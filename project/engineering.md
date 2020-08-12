1. [Nginx配置跨域请求 Access-Control-Allow-Origin *](https://segmentfault.com/a/1190000012550346)
    ```bash
    location / {  
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
        add_header Access-Control-Allow-Headers 'DNT,X-Mx-ReqToken,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization';

        if ($request_method = 'OPTIONS') {
            return 204;
        }
    } 
    ```

2. (1) [python+flask 配置https网站ssl安全认证](https://blog.csdn.net/dyingstraw/article/details/82698639)

   (2) [Flask配置Cors跨域](https://www.cnblogs.com/anxminise/p/9814326.html)
    ```bash
     pip install pyOpenSSL
     ```
     ```bash
     # 生成私钥，按照提示填写内容
    openssl genrsa -des3 -out server.key 1024

    # 生成csr文件 ，按照提示填写内容
    openssl req -new -key server.key -out server.csr

    # Remove Passphrase from key
    cp server.key server.key.org 
    openssl rsa -in server.key.org -out server.key

    # 生成crt文件，有效期1年（365天）
    openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
     ```
     ```python
    from flask import Flask    
    app = Flask(__name__)    
    app.run('0.0.0.0', debug=True, port=8001, ssl_context=('path_to_server.crt', 'path_to_server.key'))  
    ```

3. [gunicorn怎么配置https？](https://stackoverflow.com/questions/7406805/running-gunicorn-on-https/14163851)
    ```bash
    gunicorn --certfile=server.crt --keyfile=server.key test:app
    ```

4. Dockerfile 时区修改
    ```dockerfile
    FROM ubuntu:18.04

    ENV DEBIAN_FRONTEND=noninteractive

    RUN apt update && apt install tzdata -y
    RUN dpkg-reconfigure --frontend noninteractive tzdata
    RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    ```
5. list dict 去重
    ```python
    data_list = [{"a": "123", "b": "321"}, {"a": "123", "b": "321"}, {"b": "321", "a": "123"}]
    red_func = lambda x, y: x if y in x else x + [y]
    reduce(red_func, [[], ] + data_list)
    ```

6. 算法服务
 
 - (1) 自行管理
   - Flask 提供基础算法服务
   - Gunicorn 服务实例进程级别的负载
   - Supervisord 对于 Gunicorn 服务进程提供 daemon 管理
   - Nginx 作为 gateway 提供负载、安全相关功能
  
 - (2) Kubernetes 资源管理
   - 基础服务镜像。应该具备自行管理中的1~3功能
   - 服务配置yaml文件
   
 - (3) 常用微服务框架
   - Python: Flask, Tornado, Twisted
   - Java: Spring系列

7. 离线更新较大的ES索引时，CPU负载过高的原因及解决

    选择ES存储推荐系统的离线推荐结果，在更新离线推荐并写入ES时，接到了运维的告警信息，说是发现CPU负载很高。刚开始有些疑惑，更新的数据量比较大可能会使得网络开销较大，为何CPU负载这么高呢？因为业务的原因，这块的更新频率并不会很大，因此后边没有太在意，但记得有这个问题。后来在看ES的[官方文档](https://www.elastic.co/guide/cn/elasticsearch/guide/current/near-real-time.html#CO38-1)时看到了这段说明：

    **refresh_interval** 可以在既存索引上进行动态更新。 在生产环境中，当你正在建立一个大的新索引时，可以先关闭自动刷新，待开始使用该索引时，再把它们调回来：

    ```bash
    PUT /my_logs/_settings
    { "refresh_interval": -1 } 

    PUT /my_logs/_settings
    { "refresh_interval": "1s" } 
    ```
    因此在后续更新较大的ES索引时，可以采用上述的方法来做相关的优化
    
8. grok字段在线[测试](http://grokdebug.herokuapp.com/)

9. 优雅重启Gunicorn
   ```bash
   ps -ef | grep "gunicorn" | head -n 1 | awk '{print $2}' | xargs kill -HUP
   ```
   
10. Nginx健康检查

   推荐服务暂时做了主从高可用，Nginx对分发的流量服务后端做健康检查[官方文档](https://docs.nginx.com/nginx/admin-guide/load-balancer/http-health-check/)
   
11. Redis哨兵

上周出现了一次推荐服务线上异常情况，看log是Redis无法写入。继续追发现是Redis读写都是直接连接了主节点，当天Redis集群发生了主从切换，导致主节点IP变更，从而失去写入数据的权限。
后来改为哨兵模式，每次写入请求通过哨兵获取主节点，读取请求获取从节点，保证Redis读写服务的高可用。关于Redis Sentinel的一些原理和介绍可以看这篇[文章](https://baijiahao.baidu.com/s?id=1651062093310564248&wfr=spider&for=pc)，这里就不搬运了。代码：
```python
from redis.sentinel import Sentinel

class RedisSentinel:
    def __init__(self, sentinel_list, service_name, password):
        self.sentinel = Sentinel(sentinel_list, password=password)
        self.service_name = service_name

    def get_master_or_slave_conn(self, conn_type="master"):
        if conn_type == "master":
            master = self.sentinel.master_for(
                service_name=self.service_name)
            return master
        elif conn_type == "slave":
            slave = self.sentinel.slave_for(
                service_name=self.service_name)
            return slave
        else:
            raise NameError('No conn_type found for: {}'.format(conn_type))
            
class RedisUtil:
    sentinel_instance = None

    @classmethod
    def init(cls):
        if cls.sentinel_instance is None:
            sentinel_list = [("sentinel_ip1", "sentinel_port1"), ("sentinel_ip2", "sentinel_port2")]
            cls.sentinel_instance = RedisSentinel(sentinel_list, service_name="your-service-name", password="your-password")

    @classmethod
    def lpush(cls, key, values: list, ex=None):
        r = cls.sentinel_instance.get_master_or_slave_conn("master")
        r.lpush(key, *values)
        r.expire(key, ex)

    @classmethod
    def lrange(cls, key, start=0, end=-1):
        r = cls.sentinel_instance.get_master_or_slave_conn("slave")
        _res = r.lrange(key, start, end)
        _res = [_.decode('utf-8') for _ in _res]
        return _res 
```

12. Nginx对gRPC服务做负载均衡

[官方Blog](https://www.nginx.com/blog/nginx-1-13-10-grpc/)
