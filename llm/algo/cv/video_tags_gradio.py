import json

import cv2
import numpy as np
import requests
import os
import base64
from io import BytesIO
from PIL import Image

from tqdm import tqdm

import gradio as gr

from openai import AzureOpenAI


client = AzureOpenAI(
    api_key="xxx",
    api_version="2024-04-01-preview",
    azure_endpoint="xxx"
)


# 下载视频文件
def download_video(url, save_dir="download_video"):
    response = requests.get(url, stream=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file_path = os.path.join(save_dir, url.split("/")[-1])

    if os.path.exists(save_file_path):
        return save_file_path
    with open(save_file_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                f.write(chunk)
    return save_file_path


# 提取关键帧
def extract_keyframes(video_path, num_frames=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frames = []
    frame_diffs = []
    success, prev_frame = cap.read()
    frames.append(prev_frame)

    while cap.isOpened():
        try:
            success, curr_frame = cap.read()
        except:
            success, curr_frame = False, None
        if not success:
            cap.release()
            break

        frames.append(curr_frame)

        diff = cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))
        diff_sum = np.sum(diff)
        frame_diffs.append(diff_sum)
        prev_frame = curr_frame

    cap.release()

    # 按差分值从高到低选择关键帧
    if num_frames is None:
        # 根据视频时长进行推断，暂定至少平均5秒有一个
        num_frames = int(duration / 5)
    key_frame_indices = np.argsort(frame_diffs)[-num_frames:]
    key_frames = [frames[i] for i in sorted(key_frame_indices)]
    print("关键帧索引：", sorted(key_frame_indices))
    return key_frames


# 将关键帧转换为 base64 编码字符串
def frames_to_base64(frames):
    base64_frames = []
    for frame in frames:
        # 将图片转为RGB格式，以便与PIL兼容
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pil_img = Image.fromarray(frame_rgb)
        buffer = BytesIO()
        pil_img = Image.open(frame[0])
        pil_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_frames.append(img_str)
    return base64_frames

# 保存关键帧
def save_keyframes(keyframes, output_folder='keyframes'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    frame_file_paths = []
    for i, frame in enumerate(keyframes):
        save_path = os.path.join(output_folder, f'keyframe_{i}.jpg')
        cv2.imwrite(save_path, frame)
        frame_file_paths.append(save_path)
    return frame_file_paths


def llm_analyze(frames):
    prompt = """
    你是一位时装视觉分析专家，需要从社交媒体短视频中分析人物穿搭、场景、时间等关键因素，形成市场趋势洞察结论。
    标签需要专业、精简，符合小红书搜索系统可能会采用的标签体系。
    下边的图片是从视频中提取的关键帧，结合图片内容及顺序给出合适的视频标签，不超过5个.
     - 输出格式为JSON
     - 重点关注该场景下5W相关内容分析（where、who、what、why、when）
     - 标签之间的语义不要太相似，如TED演讲-演讲、服装-裤子等
     - 输出例子如下：[{"tag": "亲子", "score": 0.8, "reason": "第二个和第三个场景综合看是爸爸和儿子的互动"}, {"tag": "公园", "score": 0.5, "reason":"明显看到‘朝阳公园’"}, ...]
    """
    base64_frames = frames_to_base64(frames)
    image_content_list = []
    for base64_image in base64_frames:
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        image_content_list.append(image_content)
    final_content_list = [
                {"type": "text",
                 "text": prompt}
            ]
    final_content_list.extend(image_content_list)
    messages = [
        {
            "role": "user",
            "content": final_content_list
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-0806",
        messages=messages,
        max_tokens=4096,
        temperature=0.0,
        seed=101

    )
    content = response.choices[0].message.content.replace("```json", "").replace("```", "")
    print("raw:", content)
    res = json.loads(content)
    return res


with gr.Blocks() as demo:
    url_input = gr.Textbox(label="视频 URL")
    download_button = gr.Button("分析")

    with gr.Row():
        video_player = gr.Video(label="下载的视频")
        # analyze_button = gr.Button("分析")
        analysis_output = gr.JSON(label="分析结果")
        key_frames = gr.Gallery(label="关键场景")

    def handle_download(url):
        video_path = download_video(url)
        return video_path

    def handle_key_frames(video_path):
        gr.Info("正在进行视频预处理")
        _frames = extract_keyframes(video_path)
        _frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in _frames]
        gr.Info("视频预处理完成！开始进行视觉内容分析")
        return _frames

    def handle_analyze(keyframes):
        result = llm_analyze(frames=keyframes)
        gr.Info("分析完成！", duration=5)
        return result

    # download_button.click(handle_download, inputs=url_input, outputs=video_player)
    # analyze_button.click(handle_analyze, inputs=video_player, outputs=analysis_output)

    download_button.click(handle_download, inputs=url_input, outputs=video_player).then(
        handle_key_frames, inputs=video_player, outputs=key_frames
    ).then(
        handle_analyze, inputs=key_frames, outputs=analysis_output)

if __name__ == '__main__':
    demo.launch()
