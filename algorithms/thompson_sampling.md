最近忙于信息流推荐接口相关的工作，对于汤普森采样断断续续的做了些实验，主要在于Beta分布相关的东西。
Beta分布是一个分布族，是关于概率的概率分布，可以很好地模拟日常生活中一个人对于某个事件发生概率的认知：之前听说过这个，对它有个大概的认知 -> 先验概率，然后自己去经历 -> 实验数据，更新自己的认知 -> 后验概率。
```python
suc = 0
tot = 0
test_step = 100
total_round = 10001
mike = {'alpha': 0, 'beta': 0}
ctr = 0.3

for t_round in range(total_round):
    if random.random() > (1 - ctr):
        suc += 1
    tot += 1
    mike['alpha'] = suc
    mike['beta'] = tot - suc
    if t_round % test_step == 0 and t_round > test_step:
        print(mike)
        probs1 = [rbeta(mike['alpha'], mike['beta']) for _ in range(100)]
        # sns.kdeplot(probs1, label='Post prob {}'.format(t_round))
        sns.kdeplot(probs1)
        plt.axvline(x=ctr, ls="-.", c="black", label='Prior prob')
        plt.title('Beta post distribution evolve process')
        plt.pause(0.5)
plt.show()
```
<table><tr>
<td><img src=./assets/ts_1.png border=0></td>
<td><img src=./assets/ts_2.png border=0></td>
</tr></table>
