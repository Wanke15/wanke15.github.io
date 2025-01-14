from phi.assistant import Assistant
from phi.llm.azure import AzureOpenAIChat

import os
os.environ["AZURE_OPENAI_API_KEY"] = "xxx"
os.environ["AZURE_OPENAI_API_VERSION"] = "xxx"
os.environ["AZURE_OPENAI_ENDPOINT"] = "xxx"
os.environ["AZURE_DEPLOYMENT"] = "xxx"

# 1. 网页搜索

# from phi.tools.duckduckgo import DuckDuckGo
# web_agent = Assistant(
#     llm=AzureOpenAIChat(model=os.getenv("AZURE_DEPLOYMENT")),
#     tools=[DuckDuckGo()],
#     instructions=["Always include sources"],
#     show_tool_calls=True,
#     markdown=True,
# )
# web_agent.print_response("Tell me about OpenAI Sora?", stream=True)


# 2. 获取股票信息

from phi.tools.yfinance import YFinanceTools
agent = Assistant(
    llm=AzureOpenAIChat(model=os.getenv("AZURE_DEPLOYMENT")),
    tools=[YFinanceTools(stock_price=True)],
    show_tool_calls=True,
    markdown=True,
)
agent.print_response("What is the stock price of NVDA and TSLA")

# 3. Python agent

# from pathlib import Path
#
# from phi.assistant.python import PythonAssistant
# from phi.file.local.csv import CsvFile
#
# cwd = Path(__file__).parent.resolve()
# tmp = cwd.joinpath("tmp")
# if not tmp.exists():
#     tmp.mkdir(exist_ok=True, parents=True)
#
# python_agent = PythonAssistant(
#     llm=AzureOpenAIChat(model=os.getenv("AZURE_DEPLOYMENT")),
#     base_dir=tmp,
#     files=[
#         CsvFile(
#             path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
#             description="Contains information about movies from IMDB.",
#         )
#     ],
#     markdown=True,
#     pip_install=True,
#     show_tool_calls=True,
# )
# python_agent.print_response("What is the average rating of movies?")


