from llama_index.core.tools import FunctionTool
from tavily import TavilyClient

tavily_api_key = "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"
tavily_cli = TavilyClient(api_key=tavily_api_key)

def web_search(query: str) -> str:
    """Function to search the web and obtain information using a search query"""
    results = tavily_cli.search(query=query)
    return results

def mul_integers(a: int, b: int) -> int:
    """Function to multiply 2 integers and return an integer"""
    return a * b

def add_integers(a: int, b: int) -> int:
    """Function to add 2 integers and return an integer"""
    return a + b

def div_integers(a: int, b: int) -> float:
    """Function to add 2 integers and return a float"""
    return a / b

add_tool = FunctionTool.from_defaults(fn=add_integers)
mul_tool = FunctionTool.from_defaults(fn=mul_integers)
div_tool = FunctionTool.from_defaults(fn=div_integers)
search_tool = FunctionTool.from_defaults(fn=web_search)

def return_tool_list():
    return [add_tool, mul_tool, div_tool, search_tool]