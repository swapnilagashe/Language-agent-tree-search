import getpass
import os
from helper import *
from langchain_openai import AzureChatOpenAI



from node import Node
from tree_state import TreeState
from reflection import *
from initial_response import *




#expansion_chain = create_expansion_chain(prompt_template)
#res = expansion_chain.invoke({"input": "Write a research report on lithium pollution."})
#print(res)


print(RunnableConfig["configurable"])

initial_answer_chain= configure_initial_answer_chain(prompt_template,llm,tools)
#print(initial_answer_chain.invoke(
#    {"input": "Write a research report on lithium pollution."}
#))

expansion_chain =create_expansion_chain(prompt_template)
print(expansion_chain.invoke({"input": "Write a research report on lithium pollution."}))