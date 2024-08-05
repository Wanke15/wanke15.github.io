## 项目贡献归因
我们可以使用一种称为“Shapley值”的方法。这种方法在博弈论中用于公平地分配合作收益。具体来说，Shapley值考虑了每个参与者在所有可能的合作顺序中的边际贡献，然后取平均值

```python
from itertools import permutations

# 提升百分比
improvements = {
    'algo_1': 0.01,
    'algo_2': 0.06,
    'algo_3': 0.01,
    'algo_4': 0.01
}

# 所有参与者
players = list(improvements.keys())


# leader配置
# base随着团队成员数而变化(5, 4, 3, 2)：0.15, 0.25, 0.35, 0.45
leader_base = 0.25
leader_per_project_ratio = 0.2

s = 1.0
for v in improvements.values():
    s = s * (1 + v)
s = s - 1.0
print("总提升：", s)

# 计算累乘收益
def calculate_cumulative_improvement(order):
    improvement = 1.0
    for player in order:
        improvement *= (1 + improvements[player])
    return improvement


# 计算Shapley值
def calculate_shapley_value(players, improvements):
    shapley_values = {player: 0.0 for player in players}
    n = len(players)

    # 所有可能的顺序
    all_permutations = list(permutations(players))

    for perm in all_permutations:
        for i, player in enumerate(perm):
            # 当前玩家的边际贡献
            if i == 0:
                marginal_contribution = (1 + improvements[player]) - 1
            else:
                previous_improvement = calculate_cumulative_improvement(perm[:i])
                current_improvement = calculate_cumulative_improvement(perm[:i + 1])
                marginal_contribution = current_improvement - previous_improvement

            shapley_values[player] += marginal_contribution

    # 取平均值
    for player in shapley_values:
        shapley_values[player] /= len(all_permutations)

    return shapley_values


# 计算Shapley值
shapley_values = calculate_shapley_value(players, improvements)

# 打印结果
for player, value in shapley_values.items():
    print(f"{player} 的贡献: {value * 100:.2f}%, "
          f"最终占比: {value / s * 100 :.2f}%, "
          f"个人占比（2人制）:{value / s * (1 - leader_base) * (1 - leader_per_project_ratio) * 100 :.2f}%")

print("#"*100)

# (1) 方案或idea指导
# (2) 流程和工具优化，不能直接带来业务效果提升
# base部分：20%
ls = leader_base
for v in shapley_values.values():
    # 项目提成30%
    ls = ls + v / s * (1 - leader_base) * leader_per_project_ratio
print("Leader 最终占比：", ls)

# 总提升： 0.09211906000000014
# algo_1 的贡献: 1.04%, 最终占比: 11.29%, 个人占比（2人制）:6.78%
# algo_2 的贡献: 6.09%, 最终占比: 66.12%, 个人占比（2人制）:39.67%
# algo_3 的贡献: 1.04%, 最终占比: 11.29%, 个人占比（2人制）:6.78%
# algo_4 的贡献: 1.04%, 最终占比: 11.29%, 个人占比（2人制）:6.78%
# ####################################################################################################
# Leader 最终占比： 0.39999999999999986

```
