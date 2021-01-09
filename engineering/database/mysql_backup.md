### 1. main_backup.sh
```bash
mysqldump -uroot -proot --databases main > /data/mysql_backup/data/main-`date "+%Y_%m_%d_%H:%M:%S"`.sql
```

### 2. crontab
每天0点5分备份
```bash
5 0 * * * /bin/bash /data/mysql_backup/main_backup.sh &> /dev/null
```
