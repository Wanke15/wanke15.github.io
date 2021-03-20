由于使用了docker，top命令的user那一栏很可能显示的是root，而无法对应到人，需要根据pid找到对应的container，再找到对应的人
```bash
#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Usage: find_container.sh PID"
    exit -1
fi

PID=$1
for FILE_PATH in $(find /sys/fs/cgroup/memory/docker/ -mindepth 2 -maxdepth 2 -name cgroup.procs)
do
    if [ $(grep -E "^${PID}\$" "$FILE_PATH") ]
    then
        CONTAINER_ID=`echo "$FILE_PATH" | sed -n "s/.*\/\([a-z0-9]\+\)\/cgroup.procs$/\1/p"`
        docker ps -a --format "{{.ID}}: {{.Names}}" --filter id="${CONTAINER_ID}"
    fi
done
```
