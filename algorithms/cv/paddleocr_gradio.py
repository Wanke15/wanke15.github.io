import gradio as gr
import requests
requests.packages.urllib3.disable_warnings()

import base64
import io


# 定义一个函数来处理 OCR 识别
def ocr_recognition(image):
    # 将图像转换为 base64 编码
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 准备请求数据
    post_data = {
        "image_base64_list": [image_base64]
    }

    # 调用 OCR 接口
    try:
        resp = requests.post("http://127.0.0.1:1038/ocr", json=post_data)
        resp.raise_for_status()  # 检查请求是否成功
        result = resp.json()
        text = result["texts"][0]
        # 假设返回的结果是一个包含识别文本的列表
        return text
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# 使用 with 语句创建 Gradio 界面
with gr.Blocks() as app:
    gr.Markdown("# OCR Recognition")
    gr.Markdown("Upload an image to perform OCR recognition.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
        with gr.Column():
            text_output = gr.Textbox(label="Recognized Text")
    # 定义按钮和交互逻辑
    submit_button = gr.Button("Submit")
    submit_button.click(fn=ocr_recognition, inputs=[image_input], outputs=[text_output])


# 启动 Gradio 应用
if __name__ == '__main__':
    app.launch(server_name="0.0.0.0", server_port=1403)
    # app.launch()
