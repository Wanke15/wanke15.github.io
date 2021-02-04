1. [GRPC 负载均衡实现](https://blog.csdn.net/weixin_34102807/article/details/91455536)

2. [CLIENT-SIDE LOAD BALANCING IN GRPC JAVA](https://sultanov.dev/blog/grpc-client-side-load-balancing/)
```java
class MultiAddressNameResolverFactory extends NameResolver.Factory {

    final List<EquivalentAddressGroup> addresses;

    MultiAddressNameResolverFactory(SocketAddress... addresses) {
        this.addresses = Arrays.stream(addresses)
                .map(EquivalentAddressGroup::new)
                .collect(Collectors.toList());
    }

    public NameResolver newNameResolver(URI notUsedUri, NameResolver.Args args) {
        return new NameResolver() {
            @Override
            public String getServiceAuthority() {
                return "fakeAuthority";
            }
            public void start(Listener2 listener) {
                listener.onResult(ResolutionResult.newBuilder().setAddresses(addresses).setAttributes(Attributes.EMPTY).build());
            }
            public void shutdown() {
            }
        };
    }

    @Override
    public String getDefaultScheme() {
        return "multiaddress";
    }
}

public static void main(String[] args) {
    NameResolver.Factory nameResolverFactory = new MultiAddressNameResolverFactory(
            new InetSocketAddress("localhost", 50000),
            new InetSocketAddress("localhost", 50001),
            new InetSocketAddress("localhost", 50002)
    );

    ManagedChannel channel = ManagedChannelBuilder.forTarget("service")
            .nameResolverFactory(nameResolverFactory)
            .defaultLoadBalancingPolicy("round_robin")
            .usePlaintext()
            .build();
}
```

3. [ClientSideLoadBalancedEchoClient](https://github.com/saturnism/grpc-by-example-java/blob/master/kubernetes-lb-example/echo-client-lb-dns/src/main/java/com/example/grpc/client/ClientSideLoadBalancedEchoClient.java)
```java
/*
 * Copyright 2016 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.grpc.client;

import com.example.grpc.EchoRequest;
import com.example.grpc.EchoResponse;
import com.example.grpc.EchoServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.internal.DnsNameResolverProvider;
import io.grpc.util.RoundRobinLoadBalancerFactory;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * This example uses client side load balancing. It uses a name resolver to resolve a given service
 * name into a list of endpoints. You typically need to get the endpoints from a service registry.
 * This example uses DNS as the service registry. I.e., if a DNS entry has multiple A records,
 * each A record will used as a possible endpoint.
 *
 * Finally, this example uses a client side round robin load balancer to distributed the requests.
 */
public class ClientSideLoadBalancedEchoClient {
  private static int THREADS = 4;
  private static Random RANDOM = new Random();

  public static void main(String[] args) throws InterruptedException, UnknownHostException {
    String target = System.getenv("ECHO_SERVICE_TARGET");
    if (target == null || target.isEmpty()) {
      target = "localhost:8080";
    }
    final ManagedChannel channel = ManagedChannelBuilder.forTarget(target)
        .nameResolverFactory(new DnsNameResolverProvider())  // this is on by default
        .loadBalancerFactory(RoundRobinLoadBalancerFactory.getInstance())
        .usePlaintext(true)
        .build();

    final String self = InetAddress.getLocalHost().getHostName();

    ExecutorService executorService = Executors.newFixedThreadPool(THREADS);
    for (int i = 0; i < THREADS; i++) {
      EchoServiceGrpc.EchoServiceBlockingStub stub = EchoServiceGrpc.newBlockingStub(channel);
      executorService.submit(() -> {
        while (true) {
          EchoResponse response = stub.echo(EchoRequest.newBuilder()
              .setMessage(self + ": " + Thread.currentThread().getName())
              .build());
          System.out.println(response.getFrom() + " echoed");

          Thread.sleep(RANDOM.nextInt(700));
        }
      });
    }
  }
}
```
