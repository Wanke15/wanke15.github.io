deeplearning.ai 课程

https://learn.deeplearning.ai/courses/getting-structured-llm-output/lesson/cat89/introduction?courseName=getting-structured-llm-output
## 1. pydantic 生成 prompt
 - 定义需要的pydantic类
 - 把pydantic的json prompt添加到原本的prompt中，作为格式化输出的要求
 ```python
  from langchain_core.output_parsers import JsonOutputParser
  
  parser = JsonOutputParser(pydantic_object=XxxPydanticClass)
  instructions = parser.get_format_instructions()
  print(instructions)
  ```

## 2. instructor 结合 pydantic
retry方式。通过validate llm生成结果，不满足的重试调用llm，知道满足schema要求或达到设定的重试次数
```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

client = instructor.from_openai(OpenAI())

class UserInfo(BaseModel):
    name: str = Field(description="用户名")
    age: int = Field(description="年龄")

res = client.chat.completions.create(
    model=model,
    response_model=UserInfo,
    temperature=0.2,
    top_p=0,
    seed=101,
    messages=[
        {
            "role": "user",
            "content": "生成一个用户信息"
        }
    ]
)

print(res.model_dump_json(indent=2))

"""
{
  "name": "张伟",
  "age": 28
}
"""
```

## 3. outlines 修改 logits
本地模型inference，定制化
