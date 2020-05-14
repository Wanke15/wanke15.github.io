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
 ```bash
from flask import Flask    
app = Flask(__name__)    
app.run('0.0.0.0', debug=True, port=8001, ssl_context=('path_to_server.crt', 'path_to_server.key'))  
```

3. [Flask+gunicorn怎么使用https？](https://stackoverflow.com/questions/7406805/running-gunicorn-on-https/14163851)
```bash
gunicorn --certfile=server.crt --keyfile=server.key test:app
```
