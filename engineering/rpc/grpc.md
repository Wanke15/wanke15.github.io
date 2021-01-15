#### 1. grpc学习

训练好的TensorFlow模型自己写[gRPC服务](https://github.com/Wanke15/DeepRecommender/tree/master/grpc_service)

RESTful和gRPC性能差异:

样例数据：
```json
{
  "age": 28,
  "capital_gain": 0,
  "capital_loss": 0,
  "education": "Assoc-acdm",
  "education_num": 12,
  "gender": "Male",
  "hours_per_week": 40,
  "marital_status": "Married-civ-spouse",
  "native_country": "United-States",
  "occupation": "Protective-serv",
  "race": "White",
  "relationship": "Husband",
  "workclass": "Local-gov"
}
```

(1) RESTful
```bash
%timeit requests.post("http://localhost:3723/wide-and-deep/adult", json=demo_data1).json()
13.5 ms ± 41.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
(2) gRPC
```bash
%timeit adult = wide_and_deep_pb2.Adult(**demo_data1); stub.Predict(adult)
8.94 ms ± 72.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

#### 2. Java gRPC服务端和客户端

[链接](https://www.cnblogs.com/junjiang3/p/9096895.html)

#### 3. gRPC 压测工具
 - [ghz](https://github.com/bojand/ghz)
 - [用 ghz 对 gRPC 服务进行压测](https://farer.org/2020/02/20/grpc-load-testing-with-ghz/)

#### 4. rpc GUI 客户端
[bloomrpc](https://github.com/uw-labs/bloomrpc)
