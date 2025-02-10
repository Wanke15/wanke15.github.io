### 1. LLM client
from openai import AzureOpenAI as oa
def init_azure_openai():
    client = oa(
        azure_endpoint="xxx",
        api_key="xxx",
        api_version="xxx"
    )
    return client

# client = init_azure_openai()
# deployment_name = "xxx"

from openai import OpenAI
client = OpenAI(api_key="sk-xxx",
                base_url="https://api.deepseek.com")
deployment_name = "deepseek-chat"


### 2. Tools
import requests

def amap_encode(location):
    url = f"https://restapi.amap.com/v3/geocode/geo?address={location}&output=JSON&key=e5ff935ee8c6eca92f85dbcd98b0ddcf"
    resp = requests.get(url)
    if resp.status_code == 200:
        # print(resp.json())
        geocodes = resp.json()['geocodes']
        return geocodes[0]['location']
    else:
        return ''

# amap_encode("广州南站")

def amap_weather(location):
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={location}&key=e5ff935ee8c6eca92f85dbcd98b0ddcf"
    resp = requests.get(url)
    if resp.status_code == 200:
        # print(resp.json())
        tmp = resp.json()['lives']
        return str(tmp)
    else:
        return ''

# amap_weather("广州")


### 3. LLM function call
args = {}

messages=[
        # {"content": "东京时间", "role": "user"},
        {"content": "杭州萧山区天气", "role": "user"},
        # {"content": "杭州东站经纬度", "role": "user"}
        ]

response = client.chat.completions.create(
        model=deployment_name,
        temperature=args.get('temperature', 0.0001),
        max_tokens=args.get('max_tokens', 4095),
        top_p=args.get('top_p', 0.0001),
        seed=args.get('seed', 1213),
        frequency_penalty=args.get('frequency_penalty', 0),
        presence_penalty=args.get('presence_penalty', 0),

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "amap_weather",
                    "description": "高德地图天气查询工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市具体名称，如`北京市海淀区`请只描述为`北京市`，不要到区县级别",
                            },
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "amap_encode",
                    "description": "高德地图地理编码工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "希望查询的地理位置",
                            },
                        },
                        "required": ["location"]
                    }
                }
            },
        ],
    
        tool_choice = "auto",
        messages=messages
    )

# response


### 4. Function call
response_message = response.choices[0].message
messages.append(response_message)

print("Model's response:")  
print(response_message)  

import json

# Handle function calls
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        if tool_call.function.name == "amap_weather":
            function_args = json.loads(tool_call.function.arguments)
            print(f"Function arguments: {function_args}")  
            _response = amap_weather(
                location=function_args.get("location")
            )
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "amap_weather",
                "content": _response,
            })

        if tool_call.function.name == "amap_encode":
            function_args = json.loads(tool_call.function.arguments)
            print(f"Function arguments: {function_args}")  
            _response = amap_encode(
                location=function_args.get("location")
            )
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "amap_encode",
                "content": _response,
            })
else:
    print("No tool calls were made by the model.")  

# original messages plus function tool result
# print(messages)



### 5. LLM final response
final_response = client.chat.completions.create(
    model=deployment_name,
    messages=messages,
)
final_response

