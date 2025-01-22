
from datetime import datetime, timedelta

import requests
import time


from sqlalchemy import create_engine
import pandas as pd

web_hook = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx"


def query_impala(sql):
    URL = 'impala://xxxx'
    impala_engine = create_engine(URL)
    df = pd.read_sql(sql, impala_engine)
    return df


def date_add(base_date, days):
    # 将日期字符串转换为 datetime 对象
    date = datetime.strptime(base_date, '%Y-%m-%d')

    # 计算新的日期
    new_date = date + timedelta(days=days)

    # 输出新的日期
    return new_date.strftime('%Y-%m-%d')

def get_cur_time():
    cur_day = time.strftime("%Y-%m-%d", time.localtime())
    cur_hour = time.strftime("%H:%M:%S", time.localtime())
    return cur_day, cur_hour


def send_md_message(message, users=['all']):
    """
    发送信息
    :param message:
    :return:
    """
    md_msg_template = {"msgtype": "markdown", "markdown": {"content": ""}}

    cur_day, cur_hour = get_cur_time()

    ## 文本信息
    final_msg = message + ",".join(['<@{}>'.format(u) for u in users])
    base_mark = """{} {}\n{}""".format(cur_day, cur_hour, final_msg)

    md_msg_template["markdown"]["content"] = base_mark

    requests.post(web_hook, json=md_msg_template)


def check(tabel_names):
    cur_day, cur_time = get_cur_time()
    biz_date = date_add(cur_day, -1)

    msg = ""
    for tabel_name in tabel_names:
        if tabel_name.endswith("_feature"):
            _sql = "select count(1) as cnt from {} where dt = '{}'".format(tabel_name, biz_date)
        else:
            _sql = "select count(1) as cnt from {} where biz_date = '{}'".format(tabel_name, biz_date)

        try:
            _df = query_impala(_sql)
            cnt = _df["cnt"].values.tolist()[0]
            if cnt > 10000:
                msg += """<font color="green">库表：{}</font>，\t日期：{}，\t数据量为：{}\n""".format(tabel_name, biz_date, cnt)
            else:
                msg += """<font color="red">库表：{}</font>，\t日期：{}，查询数据较少：{}，请重点关注\n""".format(tabel_name, biz_date, cnt)
        except Exception as e:
            msg += "出错了：{}".format(e)

    print(msg)
    send_md_message(msg)


if __name__ == '__main__':
    tabel_names = ['search_algo.search_log', 'search_algo.query_feature', 'search_algo.product_feature',
                   'search_algo.product_query_cross_feature', 'search_algo.merged_label_table']
    check(tabel_names)
    
    #while True:
    #    cur_day, cur_hour = get_cur_time()
    #    if "06:45:00" < cur_hour < "07:30:00":
    #        print("当前时间：", cur_day, cur_hour)
    #        check(tabel_names)
    #        time.sleep(3600 * 2)
    #    else:
    #        time.sleep(600)

