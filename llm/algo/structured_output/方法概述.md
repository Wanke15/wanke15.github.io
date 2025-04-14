deeplearning.ai 课程

https://learn.deeplearning.ai/courses/getting-structured-llm-output/lesson/cat89/introduction?courseName=getting-structured-llm-output
## 1. pydantic 生成 prompt
把pydantic的json promp添加到原本的prompt中，作为格式化输出的要求

## 2. instructor 结合 pydantic
retry方式。通过validate llm生成结果，不满足的重试调用llm，知道满足schema要求或达到设定的重试次数

## 3. outlines 修改 logits
本地模型inference，定制化
