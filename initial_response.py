from typing import List
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError

from tree_state import TreeState
from node import Node
from helper import *


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ])
    
    

def configure_initial_answer_chain(prompt_template,llm,tools):
    
    answer_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
        run_name="GenerateInitialCandidate")
    return answer_chain

# Define the node we will add to the graph
def generate_initial_response(state: TreeState) -> dict:
    """Generate the initial candidate response."""
    parser = JsonOutputToolsParser(return_id=True)
    res = initial_answer_chain.invoke({"input": state["input"]})
    parsed = parser.invoke(res)
    tool_responses = tool_executor.batch(
        [ToolInvocation(tool=r["type"], tool_input=r["args"]) for r in parsed]
    )
    output_messages = [res] + [
        ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
        for resp, tool_call in zip(tool_responses, parsed)
    ]
    reflection = reflection_chain.invoke(
        {"input": state["input"], "candidate": output_messages}
    )
    root = Node(output_messages, reflection=reflection)
    return {
        **state,
        "root": root,
    }


    
def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
    global llm
    n = config["configurable"].get("N", 5)
    bound_kwargs = llm.bind_tools(tools=tools).kwargs
    chat_result = llm.generate(
        [messages.to_messages()],
        n=n,
        callbacks=config["callbacks"],
        run_name="GenerateCandidates",
        **bound_kwargs
    )
    return [gen.message for gen in chat_result.generations[0]]

def create_expansion_chain(prompt_template):
    expansion_chain = prompt_template | generate_candidates
    return expansion_chain

