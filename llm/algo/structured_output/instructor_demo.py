# coding:utf8
from enum import Enum
from typing import Literal

import instructor
from pydantic import BaseModel, Field
from openai import OpenAI

from openai import AzureOpenAI

raw_client = AzureOpenAI(
    api_key="xxx",
    api_version="2024-08-01-preview",
    azure_endpoint="https://xxx"
)
model = "xxx"

# raw_client = OpenAI(
#     base_url = 'http://xxx:11434/v1',
#     api_key='ollama', # required, but unused
# )
# model = "llava"

# raw_client = OpenAI(api_key="xxx",
#                 base_url="https://ark.cn-beijing.volces.com/api/v3")
# model = "xxx"

client = instructor.from_openai(raw_client)

# Define your desired output structure
class RedNoteInfo(BaseModel):
    year: list[int] = Field(description="年份", examples=[2023, 2024])
    season: list[str] = Field(description="季节", examples=["春", "夏", "秋", "冬"])
    style: list[Literal['甜酷', '酷感', '机能', '优雅知性', '摩登', '元气', '复古', '简约', '萌趣', '街头', '休闲', '户外', '运动', '通勤', '校园', '城市', '度假', '朋克', '学院', '萌趣', '田园', '简约', '芭蕾风', '韩式甜美风', '中性风', '解构风', '机能风', '美式街头', '工装风', '废土风', '篮球街潮', 'Y2k风', '多巴胺风', '田园风', '轻户外', '嘻哈风', '国潮风', '滑板街潮', '小香风', '老钱风', '山系户外', '法式风', 'City boy', 'Clean fit', '二次元风', '萌趣图案风']] = Field(description="风格")
    scene: list[str] = Field(description="场景描述，一般为在什么地方做什么事情")
    branch: list[Literal['男装', '女装', '童装', '未知']] = Field(description="服装品类分部")
    items: list[str] = Field(description="关键单品")
    color: list[str] = Field(description="关键色彩")
    shape: list[str] = Field(description="关键廓形")
    fabric: list[str] = Field(description="关键面料")
    pattern: list[str] = Field(description="关键图案")
    detail: list[str] = Field(description="关键细节")


# Extract structured data from natural language
res = client.chat.completions.create(
    model=model,
    response_model=RedNoteInfo,
    temperature=0.2,
    top_p=0,
    seed=101,
    messages=[
        {
            "role": "user",
            "content": """
            你是一个顶尖的时尚产业分析师，需要从用户提供图片或文字中完成以下结构化任务：

一、内容过滤（前置条件） 
1. 仅分析明确讨论服装/配饰趋势的内容，过滤美妆、美食等无关主题 
2. 排除广告推广性质的内容（标注#赞助 #合作等标签的帖子）
3. 无关图片或文章直接输出空JSON

二、请从文本描述或图片分析结果中识别12大核心要素（整体中包含：风格、年份、季节、品类）： 
1. 整体：图片的整体的风格、年份、季节、品类（女装、男装、男女装）。具体风格类型: ['甜酷', '酷感', '机能', '优雅知性', '摩登', '元气', '复古', '简约', '萌趣', '街头', '休闲', '户外', '运动', '通勤', '校园', '城市', '度假', '朋克', '学院', '萌趣', '田园', '简约', '芭蕾风', '韩式甜美风', '中性风', '解构风', '机能风', '美式街头', '工装风', '废土风', '篮球街潮', 'Y2k风', '多巴胺风', '田园风', '轻户外', '嘻哈风', '国潮风', '滑板街潮', '小香风', '老钱风', '山系户外', '法式风', 'City boy', 'Clean fit', '二次元风', '萌趣图案风']
2. 单品：具体的服装单品: ['T恤', '衬衫', '牛仔裤', '长裤', '裙子', '连衣裙', '外套', '夹克', '风衣', '羊毛衫', '毛衣', '卫衣', '短裤', '背心', '西装', '西裤', '礼服', '套装', '羽绒服', '棉服', '大衣', '皮夹克', '皮裤', '运动裤', '运动衫', '泳装', '睡衣', '家居服', '围巾', '手套', '帽子']
3. 色彩：精确到潘通色卡编码的色值描述（PANTONE 13-1023 TPX） 
4. 面料：具体材质和特殊处理: ['棉', '麻', '丝', '羊毛', '羊绒', '涤纶', '尼龙', '氨纶', '粘胶纤维', '人造丝', '醋酸纤维', '天丝', '牛仔布', '灯芯绒', '雪纺', '蕾丝', '缎', '乔其纱', '薄纱', '花呢', '针织面料', '毛毡', '呢绒', '绸缎', '氯纶', '莫代尔', '丝绒', '休闲布', '弹性布', '变形纤维']
5. 工艺：特殊制作工艺: ["扎染", "刺绣", "印花", "抽褶", "明线", "洗水", "手摇花", "烫银", "贴布绣", "镂空"] 
6. 图案：具象化图案描述: ["萌趣图案", "中文字", "英文数字", "甜酷图文", "潮酷图文", "新中式", "田园植物", "动漫", "美式插画", "IP联名", "条纹", "格纹", "波点"] 
7. 元素：设计细节元素: ["金属", "亮片", "流苏", "羽毛", "提花", "绑带", "木耳边", "蝴蝶结"]
8. 廓形：三维造型特征: ["A型", "H型", "X型", "O型", "T型"]
9. 场景：时尚元素的适用场景。结合在哪里、做什么这两方面信息进行归纳提取生成一个逻辑自洽的最终场景：["职场", "社交", "家庭", "运动", "海滨度假"]
注意：年份和季节必须有明确严格的文本或视觉元素依据且是描述某个时尚趋势的，不要给出推测的结果或只是某个品牌等创建的时间；年份不会小于2023

三、季节关联规则通过以下方式建立要素与季节的绑定关系： 
1. 直接声明：当出现"xxx年春夏系列"等明确季节表述时，但需要注意季节不能是类似“秋冬”这种，需要拆解为“秋”、“冬”
2. 上下文推断：分析穿搭场景描述（如"适合冬季叠穿"） 
3. 图片线索：结合用户标注的季节标签或视觉元素（需文字化描述）

四、置信度评估（为后续算法加权）为每个识别要素标注置信等级confidence： 
- 高（3级）：要素被明确命名且配有示例图 
- 中（2级）：要素被描述但无视觉佐证 
- 低（1级）：需通过类比推断得出的元素

六、特殊要求
1. 输出内容要严格按照步骤二中的补充的列表信息，不要自行扩展与补充；“轻薄风衣”不是一种风格，年份不要推测
2. 颜色必须输出潘通色号，不要用普通的文字如：深色、浅色等
3. 单品只输出核心词如：夹克、西装、衬衫，不要附加如收腰、都市、套装等描述单品时的非核心词
4. 不要输出模糊、笼统、修饰性的内容，如：奢华的深色调、超大
5. 注意：输出的风格只有一个；但其他要素可能有多个，输出为list
6. branch为品类分部，只能从女装、男装、男女装中选择一个，不能为其他输出；这个信息一般在文章标题中有说明
7. 需要输出为list的内容，如果没有贴切的结果则输出为空列表[]
每一个输出必须有理有据并严格按照上述要求，不确定的输出为“其他”，不要瞎编乱造或不合实际的推理

七、特别补充
1. 年份year不要推测。不要因为图片中没有明确的年份标注，就给出推测年份，并在year_reason中解释
2. 如果当前图片有明确的主题，如"关键细节&配饰"，则只专注于“元素”内容提取；“KEY ITEMS”代表关键单品。其他要素类似，某一个message可以只专注于某一个要素，此时其他要素不要强加提取
3. 每一个要素的输出都是准确的、有概括性的，避免使用冗余的修饰。如“魔术贴立体口袋”=》“口袋”，“羽绒夹克”=》“夹克”，“户外运动”=》“运动”


文本如下：
2025初秋：米兰趋势亮点
本季关键词：奢华深色调 / 收腰西装 / 超大格纹 秀场亮点 奢华的深色调为都市、假日风格注入新的温暖和优雅，收腰西装以优雅女性化的轮廓，注入女性力量感，超大格纹面料打造从田园到都市的不同方式转变，赋予宁静和放松。 关键色彩：奢华的深色调关键廓形：摩登收腰西装、都市夹克、衬衫套装、精致的钩针关键面料：超大格纹关键图案：水手条纹关键细节：浪漫绑带、蕾丝边缘 ﻿#流行趋势﻿ ﻿#时尚﻿ ﻿#服装品牌﻿ ﻿#米兰时装周﻿ ﻿#小红书时装周群聊﻿ ﻿#西装﻿ ﻿#夹克﻿ ﻿@时尚薯﻿ ﻿@WOW-TREND 热点趋势
            """
        }
    ],
)

print(res.model_dump_json(indent=2))

"""
{
  "year": [
    2025
  ],
  "season": [
    "秋"
  ],
  "style": [
    "优雅知性"
  ],
  "scene": [],
  "branch": [
    "女装"
  ],
  "items": [
    "西装",
    "夹克",
    "衬衫"
  ],
  "color": [],
  "shape": [],
  "fabric": [
    "格纹"
  ],
  "pattern": [
    "条纹"
  ],
  "detail": [
    "绑带",
    "蕾丝"
  ]
}
"""
