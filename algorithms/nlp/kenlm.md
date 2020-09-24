##### 1. Ubuntu 18.04 build kenlm
```Dockerfile
FROM ubuntu:18.04

RUN apt update && apt install -y build-essential libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev cmake git python3-pip vim

RUN git clone https://github.com/Wanke15/kenlm.git --depth 1

RUN cd kenlm && mkdir -p build && cd build && cmake .. && make -j 8

```
