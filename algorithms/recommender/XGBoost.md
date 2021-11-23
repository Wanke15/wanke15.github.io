1. scala-sparkML学习笔记：xgboost进行分布式训练: https://cloud.tencent.com/developer/article/1496599
2. ml.dmlc.xgboost4j.java: https://mvnrepository.com/artifact/ml.dmlc/xgboost4j
3. JAVA训练XGBOOST: https://blog.csdn.net/qq_24834541/article/details/103797256
4. xgboost4j-spark的坑
模型设置的随机种子seed数据类型应该为Long，否则在保存模型的时候(model.write.overwrite().save(savePath))会报错：
```bash
Exception in thread "main" java.lang.ClassCastException: java.lang.Integer cannot be cast to java.lang.Long at scala.runtime.BoxesRunTime.unboxToLong(BoxesRunTime.java:105)
```
