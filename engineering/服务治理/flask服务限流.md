## 1. 场景
某个时段搜索流量突增，主流程没有做限流与熔断，某个算法服务的资源不及主流程，结果就被打爆了☹，后续调查发现是一只疯狂的爬虫，踏马的

## 2. 流控
 - 接入公司层面的流控平台
 - 代码级别，[flask-limiter](https://flask-limiter.readthedocs.io/en/stable/)服务限流

## 3. demo
```python
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)


@app.route("/slow")
@limiter.limit("1 per day")
def slow():
    return ":("


@app.route("/medium")
@limiter.limit("2/second", override_defaults=False)
def medium():
    return ":|"


@app.route("/fast")
def fast():
    return ":)"


@app.route("/ping")
@limiter.exempt
def ping():
    return "PONG"


if __name__ == '__main__':
    app.run()

```

## 4. 说明
The above Flask app will have the following rate limiting characteristics:

 - Rate limiting by remote_address of the request

 - A default rate limit of 200 per day, and 50 per hour applied to all routes.

 - The slow route having an explicit rate limit decorator will bypass the default rate limit and only allow 1 request per day.

 - The medium route inherits the default limits and adds on a decorated limit of 1 request per second.

 - The ping route will be exempt from any default rate limits.
 
