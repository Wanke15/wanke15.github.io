1. Docker快速创建
```yml
version: '2'

services:
  zoo1:
    image: wurstmeister/zookeeper
    hostname: zoo1
    ports:
      - "2181:2181"
    container_name: zookeeper


  kafka1:
    image: wurstmeister/kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_HOST_NAME: localhost
      KAFKA_ZOOKEEPER_CONNECT: "zoo1:2181"
      KAFKA_BROKER_ID: 1
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_CREATE_TOPICS: "test_topic"
    depends_on:
      - zoo1
    container_name: kafka
```

2. Producer
```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    bootstrap_servers=['localhost:9092']
)
for i in range(2):
    data = {
        "name": "李四",
        "age": 23,
        "gender": "男",
        "id": i
    }
    producer.send('test_topic', data)
producer.close()
```

3. Consumer
```python
import json
from kafka import KafkaConsumer

consumer = KafkaConsumer('test_topic', bootstrap_servers='localhost:9092',
                         auto_offset_reset='latest', #
                         enable_auto_commit=True, #
                         auto_commit_interval_ms=1000, #
                         group_id='my-group', # auto_commit and group_id make sure consume one and only onece.
                         value_deserializer=json.loads)
for msg in consumer:
    print(msg.value)

```
