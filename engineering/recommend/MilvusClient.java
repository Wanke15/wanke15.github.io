package org.jeff.recall;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.LogLevel;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;

import java.util.*;

//<!-- milvus依赖 -->
//<dependency>
//<groupId>io.milvus</groupId>
//<artifactId>milvus-sdk-java</artifactId>
//<version>2.3.3</version>
//</dependency>
//
//<!-- https://mvnrepository.com/artifact/com.google.guava/guava -->
//<dependency>
//<groupId>com.google.guava</groupId>
//<artifactId>guava</artifactId>
//<version>32.1.2-jre</version>
//</dependency>
//
//<!-- https://mvnrepository.com/artifact/com.google.protobuf/protobuf-java -->
//<dependency>
//<groupId>com.google.protobuf</groupId>
//<artifactId>protobuf-java</artifactId>
//<version>3.25.1</version>
//</dependency>

public class MilvusClient {
    private static final String COLLECTION_NAME = "gul_product_vec";
    private static final String VECTOR_FIELD = "dssm_vec";
    private static final String OUT_FIELD = "product_id";
    private static final Integer VECTOR_DIM = 64;

    public static void main(String[] args) {
        // Connect to Milvus server. Replace the "localhost" and port with your Milvus server address.
        MilvusServiceClient milvusClient = new MilvusServiceClient(ConnectParam.newBuilder()
                .withHost("127.0.0.1")
                .withPort(19530)
                .build());

        // set log level, only show errors
        milvusClient.setLogLevel(LogLevel.Error);

        List<Float> vector = Arrays.asList(0.9589034362911009f,0.1865532156472709f,0.08881792986341885f,0.8462033218503091f,0.007529345615173932f,0.013473969998568203f,0.7852878326501134f,0.9502687720799752f,0.0791350817569354f,0.9338483848080281f,0.2878017066971148f,0.8795333241953691f,0.9846353924749793f,0.19668402494043025f,0.07680092653331183f,0.34756296548796617f,0.16426818136372523f,0.6208183672739735f,0.8205743315374683f,0.02627030493269289f,0.03350756886919504f,0.41326420376206086f,0.9497034038789627f,0.30307231436385296f,0.28048908397539773f,0.018028266155862616f,0.20586734631837555f,0.1806156775327623f,0.3274660452074232f,0.4382737939496075f,0.6083110237308493f,0.9293071061539742f,0.11978452325160105f,0.9692426973143953f,0.9668598254469003f,0.9701485596593371f,0.1757174893943041f,0.429836360115349f,0.13957326174045592f,0.07183365515315532f,0.4180763190956194f,0.2973858366130655f,0.734040333330003f,0.9495922349513717f,0.573669654401419f,0.6334448773072336f,0.19526529360429645f,0.023814151903019942f,0.47579208675937146f,0.5886623754186187f,0.7645739525405058f,0.30473245449562847f,0.1483183690608878f,0.0984474955060679f,0.43179396645059454f,0.20699075016583635f,0.937619884703297f,0.48156985626480453f,0.821655076506965f,0.9973812505120481f,0.11095836078367505f,0.5810179075019997f,0.5164771140222146f,0.20547606240201932f);
        R<SearchResults> searchRet = milvusClient.search(SearchParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .withMetricType(MetricType.L2)
                .withTopK(5)
                .withVectors(Arrays.asList(vector))
                .withVectorFieldName(VECTOR_FIELD)
                .withParams("{\"nprobe\": 10}")
                .addOutField(OUT_FIELD)
                .build());

        if (searchRet.getStatus() == R.Status.Success.getCode()) {
            List<String> recallProducts = new ArrayList<>(searchRet.getData().getResults().getIds().getStrId().getDataList());
            List<Float> recallProductScores = searchRet.getData().getResults().getScoresList();
            Map<String, Float> scoreMap = new HashMap<>();
            for(int i=0; i < recallProducts.size(); i++) {
                scoreMap.put(recallProducts.get(i), recallProductScores.get(i));
            }

            System.err.println(scoreMap);
        } else {
            System.err.println("milvus查询失败！" + searchRet.getException());
        }

        milvusClient.insertAsync(InsertParam.newBuilder()
                        .withCollectionName("")
                        .withDatabaseName("")
                        .withFields(new ArrayList<>())
                        .withPartitionName("")
                        .withRows(new ArrayList<>())
                .build());

    }
}
