1. hdfs-site.xml
```xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!--

<!-- Put site-specific property overrides in this file. -->

<configuration>
    <property>
        <name>dfs.permissions.enabled</name>
        <value>false</value>
    </property>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property> 
        <name>dfs.namenode.name.dir</name>
        <value>/C:/Users/jeffk/Documents/BigData/hadoop-2.7.7/namenode_dir</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/C:/Users/jeffk/Documents/BigData/hadoop-2.7.7/datanode_dir</value>
    </property>
</configuration>

```

2. 缺少winutils.exe文件
```bash
Failed to locate the winutils binary in the hadoop binary path java.io.IOException
```

3. 缺少hadoop.dll文件
```bash
org.apache.hadoop.io.nativeio.NativeIO$Windows.access0(Ljava/lang/String;I)Z
```
这两个都是在windows平台上的问题，去[这里](https://github.com/steveloughran/winutils/tree/master/hadoop-2.7.1/bin)下载对应的文件
