### 1. 安装
1. conda create -n superset python=3.10 pip
2. conda activate superset
3. pip install apache-superset==4.0.2
4. export FLASK_APP=superset
5. export SUPERSET_CONFIG_PATH=/data1/search/bi_conf/superset_config.py   在这里配置SECRET_KEY以及SQLALCHEMY_DATABASE_URI
6. superset fab create-admin --username admin --firstname Superset --lastname Admin --email admin@superset.com --password search_jeff_1024
7. superset init
8. superset run -h 0.0.0.0 -p 8088 --with-threads --reload --debugger


### 2. clickhouse 安装
```bash
docker run -dit --name=clickhouse \
-p 8123:8123 -p 9109:9009 -p 9090:9000 \
--ulimit nofile=262144:262144 \
-v /data1/clickhouse/data:/var/lib/ \
-v /data1/clickhouse/log:/var/log/ \
yandex/clickhouse-server:22.1.3.7
```

### 4. clickhouse-connect
```bash
pip install clickhouse-connect==0.7.0
```
