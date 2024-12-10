### 1. 搜索词点击次数统计
```python
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st

def query_impala(sql):
    URL = 'impala://xxxx:21050'
    impala_engine = create_engine(URL)
    df = pd.read_sql(sql, impala_engine)
    return df


df = query_impala("""
select query, count(1) as cnt 
from search_algo.search_log 
where biz_date >= '2024-09-01'
and log_type = 'click'
and query_frequency in ('高频词', '中频词')
group by query 
having count(distinct user_id) > 10
order by cnt desc 
""")

query2cnt = dict(zip(df['query'], df['cnt']))

```

### 2. Term组合生成(分词+拼音)
```python
import jieba
import re
from pypinyin import pinyin, lazy_pinyin, Style

def generate_query_combinations(query):
    res = []

    # 去除特殊符号
    query = "".join(re.findall('[\u4e00-\u9fa5]+', query, re.S))

    if not query:
        return []

    query_terms = jieba.lcut_for_search(query)
    res.extend(query_terms)

    py = lazy_pinyin(query, v_to_u=True)

    if len(py) > 1:
        res.append("".join(py))
        res.append(" ".join(py))

        res.append("".join([py[0][0], py[1][0]]))
    
        first = py[0]
        second = py[1]

        [res.append(first[0:i+1]) for i in range(len(first))]
        [res.append(first + second[0:i+1]) for i in range(len(first))]

        [res.append(query[0] + second[0:i+1]) for i in range(len(first))]

        res.append(first + query[1])
    else:
        first = py[0]
        [res.append(first[0:i]) for i in range(len(first))]
        
    return res

generate_query_combinations("钙片")

```

### 3. Trie构建
```python
import pygtrie
from tqdm import tqdm

# 初始化一个Trie
trie = pygtrie.CharTrie()

for word in tqdm(df['query']):
    freq = query2cnt[word]
    trie[word] = (word, freq)

    for g in generate_query_combinations(word):
        trie[g] = (word, freq)
```

### 4. SUG词召回和排序逻辑
（1）召回：Trie前缀匹配

（2）排序：搜索次数
```python
# 定义一个函数来获取前缀匹配的建议词
def get_suggestions(prefix, top_k=10):
    if not prefix:
        return []
    try:
        recall = list(trie.iteritems(prefix))
    except KeyError as e:
        recall = []
    recall_with_cnt = [(k[1][0], k[1][1]) for k in recall]
    rank = sorted(recall_with_cnt, key=lambda v: v[1], reverse=True)

    res = []
    keys = set()
    for i in rank:
        if len(res) >= top_k:
            break
            
        if i[0] not in keys:
            res.append(i)
            keys.add(i[0])
            
    return res

get_suggestions("gaipian")

```

### 5. Jupyter Notebook 交互可视化demo
（1）conda install -n base -c conda-forge jupyterlab_widgets

（2）conda install -n base -c conda-forge ipywidgets

（3）Restart kernel

```python
from ipywidgets import interact

@interact
def sug(key='二'):
    return get_suggestions(key)

```
