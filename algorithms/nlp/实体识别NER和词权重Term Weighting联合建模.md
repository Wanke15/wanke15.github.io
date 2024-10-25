### 1. 利用NER模型，输出实体的平均自注意力权重
```python
import torch
from transformers import BertTokenizer, BertModel, BertForTokenClassification
from transformers import pipeline

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# 使用预训练的NER模型
ner_model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
ner_pipeline = pipeline('ner', model=ner_model, tokenizer=tokenizer)

# 输入查询
query = "我要买二甲双胍，顺便看看有没有血糖仪"
inputs = tokenizer(query, return_tensors='pt')

# 获取模型输出，包括注意力权重
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
attentions = outputs.attentions

# 选择最后一层的自注意力权重
self_attention_weights = attentions[-1]

# 计算每个词的平均自注意力权重
average_self_attention_weights = self_attention_weights.mean(dim=1).squeeze()

# 使用NER模型进行实体识别
ner_results = ner_pipeline(query)

# 打印每个实体的注意力权重
for entity in ner_results:
    entity_text = entity['word']
    entity_start = entity['start']
    entity_end = entity['end']
    
    # 找到实体在tokenizer中的位置
    entity_tokens = tokenizer.tokenize(entity_text)
    entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
    entity_positions = [i for i, token_id in enumerate(inputs['input_ids'][0]) if token_id in entity_token_ids]
    
    # 计算实体的平均注意力权重
    entity_attention_weight = average_self_attention_weights[entity_positions].mean().item()
    print(f"Entity: {entity_text}, Type: {entity['entity']}, Attention Weight: {entity_attention_weight}")

```

### 2. 多任务联合建模
```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.entity_classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 实体识别
        self.importance_regressor = nn.Linear(self.bert.config.hidden_size, 1)  # 重要度预测

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        entity_logits = self.entity_classifier(sequence_output)
        importance_scores = self.importance_regressor(sequence_output)
        return entity_logits, importance_scores

# 初始化模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = MultiTaskModel()

# 输入查询
query = "我要买二甲双胍，顺便看看有没有血糖仪"
inputs = tokenizer(query, return_tensors='pt')

# 模型推理
entity_logits, importance_scores = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

# 打印每个词的实体分类和重要度分数
for token, entity_logit, importance_score in zip(tokenizer.tokenize(query), entity_logits.squeeze(), importance_scores.squeeze()):
    print(f"Token: {token}, Entity Logit: {entity_logit}, Importance Score: {importance_score.item()}")

```
