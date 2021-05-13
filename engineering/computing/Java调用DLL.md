1. JAVA调用DLL 超详细代码实战: https://blog.csdn.net/hongfei568718926/article/details/80231516
```java
package implementation;  
  
import com.sun.jna.Library;  
import com.sun.jna.Native;  
  
public interface JNATestDll extends Library {  
    JNATestDll instanceDll  = (JNATestDll)Native.loadLibrary("JNATestDLL",JNATestDll.class);  
    public int add(int a,int b);  
    public int factorial(int n);  
}


import com.sun.jna.*;

/**
 * demo
 *
 */
public class TestJNA {

    public interface nativeDLL extends Library {

        ///当前路径是在项目下，而不是bin输出目录下。
        nativeDLL INSTANCE = (nativeDLL) Native
                .loadLibrary(
                        "nativeDLL",
                        nativeDLL.class);

        public double demo();
    }

    public static void main(String[] args) {

        // TODO Auto-generated method stub
        long startTime = System.currentTimeMillis(); // 获取开始时间
        double result = nativeDLL.INSTANCE.demo();
        System.out.println(result);
        System.out.println("JNA成功调用dll！！！");
        long endTime = System.currentTimeMillis(); // 获取结束时间
        System.out.println("程序运行时间： " + (endTime - startTime) + "ms");

    }

}

```
2. 几种java调用dll的方式: https://blog.csdn.net/a491857321/article/details/51504094/
