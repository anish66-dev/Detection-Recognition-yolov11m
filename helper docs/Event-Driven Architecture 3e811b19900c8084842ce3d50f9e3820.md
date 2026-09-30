# Event-Driven Architecture

Use Redis Streams as primary inter-service communication mechanism instead of direct HTTP or RabbitMQ/Kafka

**Reason:**

- Simpler Operations
- Already using Redis
- Consumer groups
- At-least-once delivery
- Sufficient throughput

**Cons:**

- Not as scalable as Kafka

---

# Architecture Diagram

![mermaid-diagram (1).png](mermaid-diagram_(1).png)