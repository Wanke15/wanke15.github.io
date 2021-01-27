## 1. 基于docker-compose搭建简易consul数据中心
```yaml
version: '2'
services:
  consul_master:
    environment:
      CONSUL_BIND_INTERFACE: eth0
    image: consul:1.6.0
    ports:
      - "8500:8500"
    networks:
      - consul_net
    command: "agent --server=true --bootstrap-expect=3 --client=0.0.0.0 -ui"

  consul_slave1:
    environment:
      CONSUL_BIND_INTERFACE: eth0
    image: consul:1.6.0
    links:
      - consul_master
    networks:
      - consul_net
    command: "agent --server=true --client=0.0.0.0 --join consul_master"

  consul_slave2:
    environment:
      CONSUL_BIND_INTERFACE: eth0
    image: consul:1.6.0
    links:
      - consul_master
    networks:
      - consul_net
    command: "agent --server=true --client=0.0.0.0 --join consul_master"

  consul_slave3:
    environment:
      CONSUL_BIND_INTERFACE: eth0
    image: consul:1.6.0
    links:
      - consul_master
    networks:
      - consul_net
    command: "agent --server=false --client=0.0.0.0 --join consul_master"

networks:
  consul_net:
    driver: bridge

```
