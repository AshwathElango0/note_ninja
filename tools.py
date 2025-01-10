from llama_index.core.tools import FunctionTool

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

def return_tool_list():
    return [add_tool, mul_tool, div_tool]