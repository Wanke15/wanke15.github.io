1. 确认K8S版本，并下载所需镜像，使得K8S可以启动
 - images.txt
 ```bash
  k8s.gcr.io/kube-proxy:v1.14.3=gotok8s/kube-proxy:v1.14.3
  k8s.gcr.io/kube-controller-manager:v1.14.3=gotok8s/kube-controller-manager:v1.14.3
  k8s.gcr.io/kube-scheduler:v1.14.3=gotok8s/kube-scheduler:v1.14.3
  k8s.gcr.io/kube-apiserver:v1.14.3=gotok8s/kube-apiserver:v1.14.3
  k8s.gcr.io/coredns:1.3.1=gotok8s/coredns:1.3.1
  k8s.gcr.io/pause:3.1=gotok8s/pause:3.1
  k8s.gcr.io/etcd:3.3.10=gotok8s/etcd:3.3.10
 ```
  - 脚本
  ```python
  import os

with open('images.txt', 'r') as f:
    lines = [l.strip() for l in f]


for idx, line in enumerate(lines):
    print('Current: ', line)
    k8s = line.split('=')[0]
    goto = line.split('=')[1]
    command1 = f'docker pull {goto}'
    os.system(command1)

    command2 = f'docker tag {goto} {k8s}'
    os.system(command2)

    command3 = f'docker rmi {goto}'
    os.system(command3)

  ```
 
2. Dashboard 安装
 - 下载镜像
 ```bash
 docker pull gcrxio/kubernetes-dashboard-amd64:v1.10.1
 docker tag gcrxio/kubernetes-dashboard-amd64:v1.10.1 k8s.gcr.io/kubernetes-dashboard-amd64:v1.10.1
```

 - 下载yaml配置文件，并修改镜像拉去方式
 
    (1) [链接](https://raw.githubusercontent.com/kubernetes/dashboard/v1.10.1/src/deploy/recommended/kubernetes-dashboard.yaml)，保存为kubernetes-dashboard.yaml
    
    (2) imagePullPolicy: IfNotPresent
    
 - 启动Dashboard
 ```bash
 kubectl apply -f kubernetes-dashboard.yaml
 kubectl proxy
 ```
 
 - 获取Token
 ```bash
 kubectl -n kube-system describe secret default| awk '$1=="token:"{print $2}'
 ```
 
 - 访问[链接](http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/)
 <img src=https://img2020.cnblogs.com/other/946674/202007/946674-20200702174145627-1851850994.png>
