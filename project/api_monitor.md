主要关注接口的响应时间、访问流量时间序列变化、响应status、各API访问占比。后续可以继续补充

1. 整体流程:
<img src=./assets/fb-es-kibana.png>

2. 关键节点:

(1) Gunicorn access log 格式定义
```bash
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)d'
```
(2)自定义ES pipeline处理filebeat采集的日志
PUT _ingest/pipeline/feed-gunicorn-pipeline
```json
{
  "description": "feed gunicorn pipeline",
  "processors": [
    {
      "grok": {
        "field": "message",
        "patterns": [
          "%{IPV4:client_ip} (.*) \\[%{HTTPDATE:timestamp}\\] (.*) %{URIPATH:path}(.*)\\ (?:%{INT:response_status}|-) (?:%{INT:response_bytes}|-) %{INT:response_time}"
        ]
      }
    },
    {
      "convert": {
        "field": "response_time",
        "type": "integer"
      }
    }
  ]
}
```
(3) Filebeat 配置
```yaml
filebeat.inputs:
- type: log
# Change to true to enable this input configuration.
  enabled: true
# Paths that should be crawled and fetched. Glob based paths.
  paths:
    - /data/feed_recommend/log/access.log

output.elasticsearch:
  # Array of hosts to connect to.
  hosts: ["localhost:9200"]
  index: "feed-request-dev-%{+yyyy.MM.dd}"
  pipeline: "feed-gunicorn-pipeline"
setup.template:
  name: 'feed-request-dev'
  pattern: 'feed-request-dev-*'
  enabled: false
```

3. Kibana仪表盘
<img src=./assets/kibana-dashboard.png>
