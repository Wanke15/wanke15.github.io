### 1. main_backup.sh
```bash
mysqldump -uroot -proot --databases main > /data/mysql_backup/data/main-`date "+%Y_%m_%d_%H:%M:%S"`.sql
```

### 2. crontab
每天0点5分备份
```bash
5 0 * * * /bin/bash /data/mysql_backup/main_backup.sh &> /dev/null
```
每天12点5分备份
```bash
5 12 * * * /bin/bash /data/mysql_backup/main_backup.sh &> /dev/null
```
每间隔5分钟备份
```bash
*/5 * * * * /bin/bash /data/mysql_backup/main_backup.sh &> /dev/null
```
每间隔12小时备份
```bash
* */12 * * * /bin/bash /data/mysql_backup/main_backup.sh &> /dev/null
```

### 3. crontab -e
根据实际需求将类似上述命令写入crontab定时任务
