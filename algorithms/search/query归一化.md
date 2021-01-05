## 思路
 - 1. 基于历史搜索query并加上query搜索频率，构建语言模型
 - 2. 计算丢字后新query的通顺度和原始query通顺度的差值
 - 3. 对差值进行排序，取收益最大的新query，且该新query的收益必须是正向的
 
 ## 代码
  - 1. 训练语言模型
  ```bash
  # 训练
  bin/lmplz -o 3 -S 80% --text data/query_char.txt --arpa result/pupu_query.arpa
  # 模型转二进制
  bin/build_binary result/pupu_query.arpa result/pupu_query.klm
  ```
   - 2. 丢字逻辑
   ```python
    from collections import defaultdict
    from operator import itemgetter

    import kenlm

    model = kenlm.Model("./model/pupu_query.klm")


    def norm(q):
        _base_score = model.score(" ".join([_ for _ in q]))
        _score_diff = defaultdict(float)
        for idx, c in enumerate(q):
            cur_q_list = [_ for sub_id, _ in enumerate(q) if sub_id != idx]
            cur_q = " ".join(cur_q_list)
            _new_score = model.score(cur_q)
            _score_diff["".join(cur_q_list)] = _new_score - _base_score

        sort_qs = sorted(_score_diff.items(), key=itemgetter(1), reverse=True)
        res = q
        # 正向且大于一定阈值。该阈值暂时是拍的，后续可以统计正常query和xjbls的query之间的差值，做个统计经验值
        if sort_qs[0][1] > 0.1:
            res = sort_qs[0][0]
        print(q, res, _base_score, sort_qs)
        return res

    norm("红枣核")
    # 红枣核 红枣 -5.841065406799316 [('红枣', 2.2615675926208496), ('红核', -0.3050847053527832), ('枣核', -0.3975353240966797)]

    norm("苹大果")
    # 苹大果 苹果 -10.093629837036133 [('苹果', 6.781871318817139), ('大果', 5.5485334396362305), ('苹大', 1.3382768630981445)]

    norm("蟹子")
    # 蟹子 蟹 -5.335107326507568 [('蟹', 1.2722787857055664), ('子', 1.1045770645141602)]

   ```
