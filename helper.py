from typing import List
from dotenv import load_dotenv
import json
import os
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.runnables import RunnableConfig
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from tree_state import TreeState
from node import Node
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
load_dotenv()

    
def initialize_llm():
    # initialize llm
    llm = AzureChatOpenAI(
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0,
    )
    return llm

llm = initialize_llm()
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tools = [tavily_tool]
tool_executor = ToolExecutor(tools=tools)


#llm = initialize_llm()
#message = HumanMessage(
#    content="Translate this sentence from English to French. I love programming."
#)
#print(llm.invoke("ice cream"))