1. 撤销所做的修改(未add，未commit)
```bash
git restore myfile
```

2. 撤销已经add的文件
```bash
git restore --staged myfile
```

3. 回滚到已经提交过的版本

（1）回到最近的一些版本
```bash
# 上一个版本
git reset --hard HEAD^
# 上上个版本
git reset --hard HEAD^^
# 依次类推，当然也有更简洁的方法，例如回退到前100个版本
git reset --hard HEAD~100
```

（2）通过提交日志，找到指定的commit ID，然后回退到指定的commit。commit的message还是要认真写的 (:
```bash
# 显示commit log
git log

# 回退到指定的版本
git reset --hard 58c11f6014e54597a7fad3ba543bdb670c18d2d0
```
