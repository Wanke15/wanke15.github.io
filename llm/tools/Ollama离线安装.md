## ollama离线安装：
## （1）release下载
https://github.com/ollama/ollama/releases

这里需要结合具体的环境进行选择，比如当时选择的是：

https://github.com/ollama/ollama/releases/tag/v0.5.12

## （2）解压缩
```bash
# 创建并切换到目录
mkdir /data/ollama && cd /data/ollama
# 解压上述下载的release压缩包
tar -xzvf xxx.tgz
```
## （3）配置环境变量
```bash
# 自定义模型保存目录
mkdir /data/ollama/models

vim ~/.bashrc

export OLLAMA_MODELS=/data/ollama/models
export OLLAMA_HOST=0.0.0.0
export PATH=/data/ollama/bin/:$PATH

source ~/.bashrc
```
## （4）服务启动
通过守护进程或screen的方式启动服务
```bash
ollama start
```
