import sqlite3
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

DB_PATH = Path("data") / "enrolled.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # people table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            status TEXT NOT NULL, -- "authorized" or "blocklisted"
            photo_path TEXT NOT NULL
        )
    ''')
    
    # embeddings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
        )
    ''')

    # alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            similarity REAL NOT NULL,
            job_id TEXT,
            frame_num INTEGER,
            alert_type TEXT DEFAULT 'restricted_entry'
        )
    ''')
    
    # Migration: add alert_type column to existing databases that lack it
    cursor.execute("PRAGMA table_info(alerts)")
    columns = [row[1] for row in cursor.fetchall()]
    if "alert_type" not in columns:
        cursor.execute("ALTER TABLE alerts ADD COLUMN alert_type TEXT DEFAULT 'restricted_entry'")
    
    conn.commit()
    conn.close()

def add_person(name, status, photo_path, embedding):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO people (name, status, photo_path)
        VALUES (?, ?, ?)
    ''', (name, status, str(photo_path)))
    
    person_id = cursor.lastrowid
    
    cursor.execute('''
        INSERT INTO embeddings (person_id, embedding)
        VALUES (?, ?)
    ''', (person_id, sqlite3.Binary(embedding.tobytes())))
    
    conn.commit()
    conn.close()
    return person_id

def get_all_people():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, status, photo_path FROM people')
    rows = cursor.fetchall()
    conn.close()
    
    people = []
    for row in rows:
        people.append({
            "id": row[0],
            "name": row[1],
            "status": row[2],
            "photo_path": row[3]
        })
    return people

def get_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.id, p.name, p.status, e.embedding
        FROM people p
        JOIN embeddings e ON p.id = e.person_id
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        embedding_blob = row[3]
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).copy()
        results.append({
            "id": row[0],
            "name": row[1],
            "status": row[2],
            "embedding": embedding
        })
    return results


def find_match(query_embedding, threshold=0.35):
    """
    Compare query embedding against all stored embeddings.
    
    Returns:
        Best-matching dict {person_id, name, status, similarity}
        or None if no match exceeds the threshold.
    """
    enrolled = get_all_embeddings()
    if not enrolled:
        return None

    best_sim = -1.0
    best_match = None

    for entry in enrolled:
        sim = float(np.dot(query_embedding, entry["embedding"]))
        if sim > best_sim:
            best_sim = sim
            best_match = entry

    if best_sim >= threshold:
        return {
            "person_id": best_match["id"],
            "name":      best_match["name"],
            "status":    best_match["status"],
            "similarity": best_sim,
        }
    return None

def delete_person(person_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First get photo path to delete file later
    cursor.execute('SELECT photo_path FROM people WHERE id = ?', (person_id,))
    row = cursor.fetchone()
    photo_path = row[0] if row else None
    
    # Cascading delete will handle embeddings if configured, but let's be safe
    cursor.execute('DELETE FROM embeddings WHERE person_id = ?', (person_id,))
    cursor.execute('DELETE FROM people WHERE id = ?', (person_id,))
    
    conn.commit()
    conn.close()
    
    # Remove photo file if it exists
    if photo_path and os.path.exists(photo_path):
        try:
            os.remove(photo_path)
        except OSError:
            pass

def save_alert(person_name, similarity, job_id=None, frame_num=None, alert_type="restricted_entry"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO alerts (person_name, similarity, job_id, frame_num, alert_type)
        VALUES (?, ?, ?, ?, ?)
    ''', (person_name, similarity, job_id, frame_num, alert_type))
    
    alert_id = cursor.lastrowid
    
    # Fetch it right back to get the formatted timestamp
    cursor.execute('SELECT timestamp FROM alerts WHERE id = ?', (alert_id,))
    row = cursor.fetchone()
    timestamp = row[0] if row else ""
    
    conn.commit()
    conn.close()
    
    return {
        "id": alert_id,
        "person_name": person_name,
        "timestamp": timestamp,
        "similarity": similarity,
        "job_id": job_id,
        "frame_num": frame_num,
        "alert_type": alert_type
    }

def get_recent_alerts(limit=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, person_name, timestamp, similarity, job_id, frame_num, alert_type
        FROM alerts
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    alerts = []
    for row in rows:
        alerts.append({
            "id": row[0],
            "person_name": row[1],
            "timestamp": row[2],
            "similarity": row[3],
            "job_id": row[4],
            "frame_num": row[5],
            "alert_type": row[6] if len(row) > 6 else "restricted_entry"
        })
    return alerts

def clear_alerts():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM alerts')
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()
