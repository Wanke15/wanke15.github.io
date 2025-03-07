from copy import deepcopy
from functools import partial

import gradio as gr
import pandas as pd


# 字段名分别为：图片URL, 类别, 属性, 原因
df = pd.read_excel("./output/xxxxx.xlsx")

def category_stats(category):
    df1 = df[df['类别'] == category]["属性"].value_counts()
    df1 = pd.DataFrame({"属性": df1.index, "数量": df1.values}).head(10)
    return df1

# 根据筛选条件过滤图片
def display_image(*selection):
    selected_df = deepcopy(df)
    sub_set_list = []
    selection = [_ for _ in selection if _]
    if selection:
        for s in selection:
            sub_set_list.append(selected_df[selected_df["属性"] == s]["图片URL"].unique().tolist())
        if sub_set_list:
            sets = [set(lst) for lst in sub_set_list]
            result = [i for i in set.intersection(*sets)]
        else:
            result = []
    else:
        # result = selected_df[selected_df["属性"] == "黑色"]["图片URL"].unique().tolist()
        result = []

    return result

# 自定义 CSS 隐藏Gradio底部信息
custom_css = """
footer {visibility: hidden;}
.gradio-container .footer {display: none !important;}
"""
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    gr.Markdown("## VOC-六要素解析")

    cats = df["类别"].unique().tolist()
    col_len = 3
    with gr.Row():
        for c in cats[:col_len]:
            gr.BarPlot(value=partial(category_stats, category=c), x="属性", y="数量", title=c, sort="-y")
    with gr.Row():
        for c in cats[col_len:2*col_len]:
            gr.BarPlot(value=partial(category_stats, category=c), x="属性", y="数量", title=c, sort="-y")

    # 筛选条件
    options = []
    select_sector = gr.Row()
    with select_sector:
        for c in cats:
            x = df[df["类别"] == c]["属性"].value_counts().index.tolist()
            x.insert(0, "")
            if c == "色彩":
                dd = gr.Dropdown(value=x[1], label=c, choices=x)
            else:
                dd = gr.Dropdown(label=c, choices=x)
            options.append(dd)

    # 展示筛选的图片
    gallery = gr.Gallery(value=display_image, inputs=options, label="图片展示", columns=5)

    # 选中图片后展示图片详情数据
    select_df = gr.DataFrame(label="详细信息")
    def on_select(evt: gr.SelectData):
        img_url = evt.value["image"]["path"]
        select_data = df[df["图片URL"] == img_url]
        return select_data

    gallery.select(on_select, None, select_df)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
    # app.launch()
