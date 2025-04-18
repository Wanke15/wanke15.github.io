import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def predict(messages, model, tokenizer):
    device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

model_dir = '/data/llm/modelscope/Qwen/Qwen2___5-0___5B-Instruct'
sft_model_save_dir = "output/model/Qwen25_05b_fudan_news/checkpoint-12/"

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)

# 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id=sft_model_save_dir)

test_texts = {
    'instruction': "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
    'input': """
文本:【 文献号 】2-371\t【原文出处】咨询与决策\t【原刊地名】武汉\t【原刊期号】200005\t【分 类 号】F61\t【分 类 名】财政与税务\t【复印期号】200007\t【 标  题 】新经济时代的税务对策：管理创新\t【 作  者 】周星/山路\t【 正  文 】\t今年以来，“新经济”一词频繁见诸报端。那么，什么是新经济？前不久，国务院发展研究中心主任王梦奎指出，新经济主要是指经济全球化和高新技术的发展，而新经济说到底是一个科技进步的问题。当前，我国发展新经济的主要任务：一是努力发展高新技术产业；二是用高新技术改造传统产业；三是经济的可持续发展。新经济的核心是创新。首先是观念的变革和创新；其次是运行模式的创新，包括制度和组织结构等；三是技术创新。而人是进行和保证创新的主体。税务部门是政府组织收入的职能部门，面对新的经济形势，必须进行管理创新，这是迎接新经济时代到来的实现需要。税务部门的管理创新，能够在现有税源基础上运用知识创新管理，找到增收的合理区间，提高税收效率、降低税收成本，增加国家财力。笔者认为，税务部门迎接新经济时代到来，进行管理创新，可以从三个方面采取对策，稳步推进。一、增加管理的知识含量及智慧资本，构建知识型税务管理进入新的世纪，国家税务部门发展的趋势可能是智慧资本成为税务部门发展的关键，税务部门的知识含量将愈来愈高。税务部门在日趋竞争激烈的经济环境中，要积极参与社会竞争，调节好收入分配关系，促进资源合理配置，推动经济良性运转及社会全面进步，这不仅取决于税务部门投入管理过程中的人、财、物等有形资产，而且更取决于税务研究开发，并以快捷的方式将研究成果融入税务管理的能力。决定税务部门竞争优势的关键因素，将从传统的机构、人员数量、税收规模转化为对税务管理知识开发、创新与有效运用的程度，这是税务部门适应新经济时代的基本要求。为此，我国税务部门应不断提高管理的知识含量与知识内容，加强知识管理势在必行。知识管理，既是对税收拥有的“智慧资源”进行有效管理的活动，也是对税务管理过程的知识变革与创新。知识管理的目的之一，是为税务部门实现显性税务知识和隐性税务知识的经济价值提供可能的途径。各级税务部门应进行有效的制度安排，进行大量的知识创新，使知识管理与税务运行协调统一，使知识管理成为税务征收管理的重要组成部分。这是税务部门在现有征管水平基础上的发展与创新。虽然税务部门的人力资源总是相对有限的，通过有限的人力资源产生的征管效益也将是有限的，但税务部门能够获得、开发的知识却是无限的，因此知识管理将给税务部门带来源源不断的征管效益，能够推动税务部门征管效益的可持续发展。二、顺应时代潮流，引进网络经济时代普遍采用的数字化管理数字化管理是指利用计算机、通信、网络、人工智能等技术，量化管理对象与管理行为，实现计划、组织、协调、服务、创新等职能的管理活动与管理方法的总称。税务部门实现数字化管理并不十分遥远，而在发达国家已经实现。数字化管理，将彻底改变税务部门征管方式，传统意义上的税收征管模式将被现代管理方法所取代，电话纳税服务、手机纳税服务、网上纳税服务等远程服务将改变我国税务机关内部的部门、岗位等组织边界。税务部门的数字管理，将具有定量化、智能化、综合化、集成化、系统化的特点。推进数字化管理，要求我国税务部门将内部的知识资源、信息资源、人力资源及外部纳税人资源数字化，并通过网络进行管理；要求运用量化管理技术来解决税务部门的管理问题。为此，税务部门应将数字化管理列为税务征管战略之一，并建立支持数字化管理的组织体系和组织形式。这样，我国税务部门传统的“金字塔型”的组织结构将逐步被“扁平型”的组织形式所取代。各级税务部门还应积极创造有利于数字化管理的税务文化，如良好的职业道德、敬业精神、征纳友谊、部门形象等。三、重视“税务人”在税务征管中的地位与作用，实行人本管理我国税务部门特别是直接从事征收管理的部门，在税收管理工作中带有较强的行政式管理倾向，重纪律、重服从，而在一定程度上忽视了职工个人的主动性、积极性与创造性，忽视了人力资源的充分开发与利用。与国外发达地区的税务部门相比，差距首先体现在对税务部门中的“人”如何进行价值判断、是否把人的因素作为管理中事实上的首要因素和本质因素上，即是否推行了“人本管理”。税务部门中的专门人才及其所掌握的税务服务知识与技能，是税务部门智慧资本的重要载体，因而人本管理以及由此调动的税务人才的主动性、积极性与创造性，是迎接新经济时代到来，税企征纳双方共同繁荣发展的根本所在。税务部门的人本管理，应在改善激励制度、改进领导方式、提高生活质量、促进工作丰富化、加强税务部门与职工及职工之间的沟通和交互作用等方面下功夫。首先建立起“尊重人、爱护人”的机制与氛围，使职工有个融洽、宽松的工作环境；其次应加大人力资源的激励与开发利用力度，对优秀人力采取教育、培养及促进个性的充分发挥等策略；第三，加强税务文化建设，既要着眼于表层的行为文化开展，又要着眼于深层的制度文化、精神文化的开发，从而提高税务职工文化素养，优化税务文化氛围，为人本管理创造条件。笔者认为，知识型税务管理，数字化管理以及人本管理等三个方面的协调运作，能够将税务管理创新推向一个新的水平，更好地服务于新经济时代。,类型选型:['Electronics', 'Enviornment', 'Military', 'Transport', 'Education', 'Literature', 'Energy', 'Art', 'Economy', 'Computer', 'Sports', 'Mine', 'Medical', 'History', 'Space', 'Communication', 'Law', 'Politics', 'Agriculture']             
"""
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

response = predict(messages, model, tokenizer)
print(response)
