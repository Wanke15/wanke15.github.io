import base64
import io

import fitz
import requests

# pdf_path = r"C:\Users\xxx\xxx.pdf"
pdf_path = r"https://arxiv.org/pdf/1601.00770.pdf"


def pdf_to_image_list(pdf_path, zoom_x=2.0, zoom_y=2.0):
    """
    将PDF转换为图片。

    参数:
    - pdf_path: PDF文件路径。
    - output_folder: 输出图片存放的文件夹。
    - zoom_x: x轴缩放比例，默认为2.0，可以根据需要调整。
    - zoom_y: y轴缩放比例，默认为2.0，可以根据需要调整。
    """
    # 打开PDF文件
    if pdf_path.startswith("http"):
        pdf_document = fitz.open(stream=io.BytesIO(requests.get(pdf_path).content), filetype="pdf")
    else:
        pdf_document = fitz.open(pdf_path)

    image_base64_list = []
    for page_num in range(len(pdf_document))[:1]:
        # 获取某一页
        page = pdf_document.load_page(page_num)

        # 设置缩放比例
        mat = fitz.Matrix(zoom_x, zoom_y)

        # 将页面转换为图片
        pix = page.get_pixmap(matrix=mat)

        img_base64 = base64.b64encode(pix.tobytes()).decode('utf-8')
        image_base64_list.append(img_base64)
        print(img_base64)

        # 保存图片，文件名为“页码数+1”.png，因为页码是从0开始的
        # output_path = f"{output_folder}/page_{page_num + 1}.png"
        # pix.save(output_path)

        return image_base64_list


pdf_to_image_list(pdf_path)
