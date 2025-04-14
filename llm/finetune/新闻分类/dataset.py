import json
from modelscope import MsDataset
# modelscope==1.25.0
# datasets==2.16.0
from tqdm import tqdm


def dataset_jsonl_transfer(subset_name, save_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []
    # 读取旧的JSONL文件
    if subset_name == 'train':
        dataset = MsDataset.load('swift/zh_cls_fudan-news', split='train')
    else:
        dataset = MsDataset.load('swift/zh_cls_fudan-news', subset_name='test', split='test')
    for data in tqdm(dataset):
        # 解析每一行的json数据
        context = data["text"]
        catagory = data["category"]
        label = data["output"]
        message = {
            "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
            "input": f"文本:{context},类型选型:{catagory}",
            "output": label,
        }
        messages.append(message)

    # 保存重构后的JSONL文件
    with open(save_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


dataset_jsonl_transfer("train", "output/data/train.jsonl")
dataset_jsonl_transfer("test", "output/data/test.jsonl")
