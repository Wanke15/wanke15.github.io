```python
# 自定义 CSS 隐藏底部信息
custom_css = """
footer {visibility: hidden;}
.gradio-container .footer {display: none !important;}
"""
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    gr.Markdown("## xxxxx")

# app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
```
