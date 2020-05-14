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

2. [python+flask 配置https网站ssl安全认证](https://blog.csdn.net/dyingstraw/article/details/82698639)

3. [Flask+gunicorn怎么使用https？](https://stackoverflow.com/questions/7406805/running-gunicorn-on-https/14163851)
```bash
gunicorn --certfile=server.crt --keyfile=server.key test:app
```
