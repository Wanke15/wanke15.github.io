1. 创建用户画像表
```sh
create 'feed_user_profile', 'basic', 'rent_car', 'feed'
```
create 之后依次为: 表名 -> 'feed_user_profile', 三个列族 -> 'basic', 'rent_car', 'feed'

2. 写入数据
```sh
put 'feed_user_profile', 'uhdiaoe8cbb00j-ijc_2020012', 'basic:age', '20'
```
put 之后依次为： 表名 -> 'feed_user_profile', row_key -> 'uhdiaoe8cbb00j-ijc_2020012', basic列族下age字段设置为20
