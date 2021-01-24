## 1. gRPC多进程 nginx负载均衡配置
```conf
worker_processes  4;

events {
    worker_connections  1024;
}


http {
    log_format  main  '$remote_addr-[$time_local]-$request-$status-$request_time';
    access_log /data/logs/access.log main;

    keepalive_timeout  90;

    upstream backend {
        server 127.0.0.1:50051;
        server 127.0.0.1:50052;
        server 127.0.0.1:50053;
        server 127.0.0.1:50054;

    }
    server {
        listen  10051  http2;
        location / {
            grpc_pass grpc://backend;
        }
    }
    include servers/*;
}
```
## 2. 监控文件变化通用模块
```python
#!/usr/bin/env python

'''
Python-Tail - Unix tail follow implementation in Python.
python-tail can be used to monitor changes to a file.
Example:
    import tail
    # Create a tail instance
    t = tail.Tail('file-to-be-followed')
    # Register a callback function to be called when a new line is found in the followed file.
    # If no callback function is registerd, new lines would be printed to standard out.
    t.register_callback(callback_function)
    # Follow the file with 5 seconds as sleep time between iterations.
    # If sleep time is not provided 1 second is used as the default time.
    t.follow(s=5) '''

# Author - Kasun Herath <kasunh01 at gmail.com>
# Source - https://github.com/kasun/python-tail

import os
import sys
import time


class Tail(object):
    ''' Represents a tail command. '''

    def __init__(self, tailed_file):
        ''' Initiate a Tail instance.
            Check for file validity, assigns callback function to standard out.

            Arguments:
                tailed_file - File to be followed. '''

        self.check_file_validity(tailed_file)
        self.tailed_file = tailed_file
        self.callback = sys.stdout.write

    def follow(self, s=1):
        ''' Do a tail follow. If a callback function is registered it is called with every new line.
        Else printed to standard out.

        Arguments:
            s - Number of seconds to wait between each iteration; Defaults to 1. '''

        with open(self.tailed_file) as file_:
            # Go to the end of file
            file_.seek(0, 2)
            while True:
                curr_position = file_.tell()
                line = file_.readline()
                if not line:
                    file_.seek(curr_position)
                    time.sleep(s)
                else:
                    self.callback(line)

    def register_callback(self, func):
        ''' Overrides default callback function to provided function. '''
        self.callback = func

    def check_file_validity(self, file_):
        ''' Check whether the a given file exists, readable and is a file '''
        if not os.access(file_, os.F_OK):
            raise TailError("File '%s' does not exist" % (file_))
        if not os.access(file_, os.R_OK):
            raise TailError("File '%s' not readable" % (file_))
        if os.path.isdir(file_):
            raise TailError("File '%s' is a directory" % (file_))


class TailError(Exception):
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message

```

## 3. metrics Monitor
```python
# encoding: utf-8
from prometheus_client import Counter, Gauge, Summary
from prometheus_client.core import CollectorRegistry
from prometheus_client.exposition import choose_encoder


class Monitor:
    def __init__(self):
        # 注册收集器&最大耗时map
        self.collector_registry = CollectorRegistry(auto_describe=False)
        self.request_time_max_map = {}

        # 接口调用summary统计
        self.http_request_summary = Summary(name="http_server_requests_seconds",
                                       documentation="Num of request time summary",
                                       labelnames=("source","method", "code", "uri"),
                                       registry=self.collector_registry)
        # 接口最大耗时统计
        self.http_request_max_cost = Gauge(name="http_server_requests_seconds_max",
                                      documentation="Number of request max cost",
                                      labelnames=("source","method", "code", "uri"),
                                      registry=self.collector_registry)

    # 获取/metrics结果
    def get_prometheus_metrics_info(self):
        #encoder, content_type = choose_encoder(handler.request.headers.get('accept'))
        encoder, content_type = choose_encoder("")
        self.reset_request_time_max_map()
        return(encoder(self.collector_registry))

    # summary统计
    def set_prometheus_request_summary(self, source, method, status, path, cost_time ):
        self.http_request_summary.labels(source,method,status,path).observe(cost_time)
        self.set_prometheus_request_max_cost(source, method, status, path, cost_time)

    # 最大耗时统计
    def set_prometheus_request_max_cost(self, source, method, status, path, cost_time):
        if self.check_request_time_max_map(path, cost_time):
            self.http_request_max_cost.labels(source,method,status,path).set(cost_time)
            self.request_time_max_map[path] = cost_time

    # 校验是否赋值最大耗时map
    def check_request_time_max_map(self, uri, cost):
        if uri not in self.request_time_max_map:
            return True
        if self.request_time_max_map[uri] < cost:
            return True
        return False

    # 重置最大耗时map
    def reset_request_time_max_map(self):
        for key in self.request_time_max_map:
            self.request_time_max_map[key] = 0.0


if __name__ == '__main__':
    a = ["127.0.0.1","Get","200","/hello_word",1.8]
    b = ["127.0.0.2","Post","200","/hello",2.2]
    g_monitor = Monitor()
    g_monitor.set_prometheus_request_summary(*a)
    g_monitor.set_prometheus_request_summary(*b)
    print(g_monitor.get_prometheus_metrics_info())
    g_monitor.set_prometheus_request_summary(*a)
    g_monitor.set_prometheus_request_summary(*a)
    print(g_monitor.get_prometheus_metrics_info())
```

## 4. Nginx日志监听，更新、获取metrics
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import threading

from monitor import Monitor
from logger import logger
from tail import Tail
base_dir = '/data/'


class NginxMetrics:
    instance = None

    @classmethod
    def init(cls, nginx_access_log=None):
        if nginx_access_log is None:
            if not os.path.exists(os.path.join(base_dir, 'logs')):
                os.mkdir(os.path.join(base_dir, 'logs'))
            nginx_access_log = os.path.join(base_dir, 'logs/access.log')
        if cls.instance is None:
            cls.instance = NginxMetrics(nginx_access_log)

    def __init__(self, nginx_access_log):
        self.monitor = Monitor()

        self.tailor = Tail(nginx_access_log)
        self.tailor.register_callback(self.update_monitor)
        # 启动监控线程
        threading.Thread(target=self.tailor.follow).start()

    @staticmethod
    def parse_log(line):
        _splits = line.strip().split('-')
        source = _splits[0]
        _request_info = _splits[2].split()
        method = _request_info[0]
        status_code = _splits[3]
        api_path = "/{}".format(_request_info[1].split('/')[-1])
        request_time = float(_splits[4]) * 1000
        _metrics = [source, method, status_code, api_path, request_time]
        return _metrics

    def update_monitor(self, line):
        try:
            metrics = self.parse_log(line)
            self.monitor.set_prometheus_request_summary(*metrics)
        except Exception as e:
            logger.error("Parse Nginx access_log error => {}".format(e))
            pass

    def get_metrics(self):
        return self.monitor.get_prometheus_metrics_info()


if __name__ == '__main__':
    NginxMetrics.init()

```

## 5. Flask blueprint
```python
NginxMetrics.init()

metrics_blueprint = Blueprint("metrics", __name__)


@metrics_blueprint.route("/metrics")
def metrics():
    """
    Prometheus metrics api
    :return:
    """
    return Response(NginxMetrics.instance.get_metrics(), mimetype="text/plain")
```
