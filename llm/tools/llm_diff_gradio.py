import random
import time
import pandas as pd
import gradio as gr

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)

file_path = "./xxxxx.xlsx"

def read_df():
    df = pd.read_excel(file_path)

    if "score" not in df.columns:
        df["score"] = ""
    if "state" not in df.columns:
        df["state"] = 0
    if "score_time" not in df.columns:
        df["score_time"] = ""

    df = df.sort_values(by="state", ascending=True)

    note_ids = df["note_id"].tolist()

    return df, note_ids


def get_note_details(df, note_id):
    note = df[df["note_id"] == note_id].reset_index()
    note_url = note["note_url"][0]
    note_text = note["text"][0]
    score_status = "已打分" if note["state"][0] else "未打分"
    score = note["score"][0]
    return note_url, note_text, score_status, score


def single_render(note, field, provider):
    return note[field + "_" + provider][0].replace("nan, ", "").replace("nan", "").replace("{, ", "{").replace("{, }",
                                                                                                               "{}")


def page_render(note, provider):
    return {
        "5W场景": single_render(note, "full_scene_list", provider),
        "WHAT": single_render(note, "what_list", provider),
        "WHERE": single_render(note, "where_list", provider),
        "WHEN": single_render(note, "when_list", provider),
        "WHO": single_render(note, "with_whom_list", provider),
        "WHY": single_render(note, "why_list", provider),
        "左脑": single_render(note, "left_brain", provider),
        "右脑情绪": single_render(note, "right_brain_emo", provider),
        "右脑IP": single_render(note, "right_brain_ip", provider),
    }

def save_df(df, file_path):
    try:
        df.to_excel(file_path, index=False)
        print("save success")
    except Exception as e:
        print("ERROR", e)
        gr.Error("评测状态保存失败：{}".format(e))

def update_score(note_id, diff):
    if diff != '':
        df.loc[df["note_id"] == note_id, "score"] = diff
        df.loc[df["note_id"] == note_id, "state"] = 1
        df.loc[df["note_id"] == note_id, "score_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # df.to_excel(file_path, index=False)
        executor.submit(save_df, df, file_path)
    remaining = df.shape[0] - sum(df["state"])
    return remaining


def display(note_id):
    note_url, note_text, score_status, score = get_note_details(df, note_id)
    note = df[df["note_id"] == note_id].reset_index()
    gpt4o_render = page_render(note, "4o")
    deepseek_render = page_render(note, "ds")
    remaining = df.shape[0] - sum(df["state"])
    return note_url, note_text, score_status, score, gpt4o_render, deepseek_render, remaining


def submit_score(note_id, diff):
    remaining = update_score(note_id, diff)
    next_note_ids = df[df["state"] == 0]["note_id"].tolist()

    if next_note_ids:
        # next_note_id = next_note_ids[0]
        next_note_id = random.choice(next_note_ids)
        print("len", len(next_note_ids))
        return display(next_note_id) + (next_note_id, "")
    else:
        gr.Info("已全部评测完！辛苦了")
        return ["", "", "", "", {}, {}, remaining, note_id, ""]

def plot_stats_bar():
    df1 = df['score'].value_counts().sort_values(ascending=False)
    df1 = pd.DataFrame({"score": df1.index, "count": df1.values})
    # print(time.localtime(), df1)
    return df1


with gr.Blocks() as demo:
    df, note_ids = read_df()
    with gr.Row():
        with gr.Column():
            note_id = gr.Dropdown(elem_id="note_id_select", label="请选择帖子ID", choices=note_ids)
            note_url = gr.Markdown()
            note_text = gr.Markdown()
            score_status = gr.Textbox(label="当前帖子打分状态及分数")
            score = gr.Textbox(label="分数")

            diff = gr.Dropdown(value="", label="GPT4o vs. DeepSeek", choices=["", "打平", "4o更好", "DS更好", "都不行"])
            remaining = gr.Textbox(label="当前剩余未打分文档数量")

            # stats_plot = gr.Button("更新汇总统计图")
            # bar_plot = gr.BarPlot(value=pd.DataFrame({"score": [], "count": []}), x="score", y="count")
            bar_plot = gr.BarPlot(value=plot_stats_bar, x="score", y="count", every=5)


        with gr.Column():
            gr.Markdown("### GPT-4O")
            gpt4o_render = gr.JSON()
        with gr.Column():
            gr.Markdown("### DeepSeek-V3")
            deepseek_render = gr.JSON()

    note_id.change(display, inputs=[note_id],
                   outputs=[note_url, note_text, score_status, score, gpt4o_render, deepseek_render, remaining])
    # submit.click(submit_score, inputs=[note_id, diff],
    #              outputs=[note_url, note_text, score_status, score, gpt4o_render, deepseek_render, remaining, note_id])

    diff.change(submit_score, inputs=[note_id, diff],
                 outputs=[note_url, note_text, score_status, score, gpt4o_render, deepseek_render, remaining, note_id, diff])

    # stats_plot.click(plot_stats_bar, [], bar_plot)

demo.launch(server_name="0.0.0.0", server_port=8501)
