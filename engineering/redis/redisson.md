### 1. 批量查询
```java
import org.redisson.Redisson;
import org.redisson.api.RedissonClient;
import org.redisson.client.codec.StringCodec;
import org.redisson.config.Config;

import java.util.Map;


public class RedissionDemo {
    public static void main(String[] args) {
        Config config = new Config();
        config.useSingleServer()
                .setAddress("redis://127.0.0.1:6379");
//        config.useSentinelServers()
//                .addSentinelAddress("redis://***:26379")
//                .addSentinelAddress("redis://***:26379")
//                .addSentinelAddress("redis://***:26379")
//                .setMasterName("masterNmae")
//                .setPassword("passWord");

        RedissonClient redisson = Redisson.create(config);
        StringCodec stringCodec = new StringCodec();

//        RBucket<String> bucket = redisson.getBucket("hello", stringCodec);
//        System.err.println(bucket.get());
//        bucket.set("world");
//        bucket.delete();

        redisson.getBucket("hello1").set("world1");
        redisson.getBucket("hello2").set("world2");
        
        // 批量查询
        Map<String, String> productFeatures = redisson.getBuckets().get("hello1", "hello2");
        for(Map.Entry<String, String> entry: productFeatures.entrySet()) {
            System.out.println(entry.getKey() + " --- " + entry.getValue());
        }
}
            
```
