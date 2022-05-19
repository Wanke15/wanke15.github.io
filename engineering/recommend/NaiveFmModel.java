/**
 * @author wangke
 * @version 1.0
 * @date 2022/5/18 3:17 下午
 */
public class NaiveFmModel {

    float sigmoid(float val) {
        return (float) (1.0 / (1.0 + Math.exp(-val)));
    }

    float[][] factors = new float[][] {
            {-0.11845343057966201f, -0.128103126655331f, -0.12515688603786665f},
            {-0.0461855438264623f, 0.00334141008463275f, 0.07812878096995332f},
            {-0.12804740862909733f, -0.12026609557880016f, -0.11485173480056374f},
            {-0.12470980244830876f, -0.12803173709417073f, -0.11880307016121176f}
    };

    float[] linearWeight = new float[] {-0.12226160429501477f, -0.10521983183258769f, 0.08552796118362704f, 0.08819288177147111f};

    float intercept = -0.1138868819483221f;

    int featureLen = linearWeight.length;
    int factorSize = factors[0].length;

    float fmInferRaw(float[] feature) {

        // linear part
        float rawPrediction = intercept;
        for (int i=0; i < featureLen; i++) {
            float val = feature[i];
            if (val != 0.0) {
                rawPrediction += val * linearWeight[i];
            }
        }

        // feature cross part
        for (int j=0; j < factorSize; j++) {
            float sumSquare = 0.0f;
            float sum = 0.0f;

            for (int i=0; i < featureLen; i++) {
                float vx = factors[i][j] * feature[i];
                sumSquare += vx * vx;
                sum += vx;
            }

            rawPrediction += 0.5 * (sum * sum - sumSquare);
        }

        return rawPrediction;
    }

    float fmInferProb(float[] feature) {
        float raw = fmInferRaw(feature);
        return sigmoid(raw);
    }


    public static void main(String[] args) {
    // Factors: -0.11845343057966201  -0.128103126655331    -0.12515688603786665
    //-0.0461855438264623   0.00334141008463275   0.07812878096995332
    //-0.12804740862909733  -0.12026609557880016  -0.11485173480056374
    //-0.12470980244830876  -0.12803173709417073  -0.11880307016121176   Linear: [-0.12226160429501477,-0.10521983183258769,0.08552796118362704,0.08819288177147111] Intercept: -0.1138868819483221

    // |4.3|3.0|1.1|0.1|0    |[4.3,3.0,1.1,0.1]|[0.6887103107760986,-0.6887103107760986]|[0.6656799675796927,0.3343200324203072] |0.0       |

    NaiveFmModel naiveFmModel = new NaiveFmModel();


    float[] feature = new float[] {4.3f, 3.0f, 1.1f, 0.1f};

    float rawPrediction = naiveFmModel.fmInferRaw(feature);
    float prob = naiveFmModel.fmInferProb(feature);

    System.out.println(rawPrediction);
    // -0.6887103107760986

    System.out.println(naiveFmModel.sigmoid(rawPrediction));
    // 0.3343200324203072

    System.out.println(prob);
    // 0.3343200324203072

}
}
