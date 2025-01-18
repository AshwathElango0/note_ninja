from llama_index.core.tools import FunctionTool
from tavily import TavilyClient
import io
import math
import contextlib

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

def execute_code(code: str) -> dict:
    """
    Executes Python code and returns the result or error.
    
    Parameters:
        code (str): The Python code to execute.
    
    Returns:
        dict: A dictionary with 'success', 'output', and 'error' keys.
    """
    # Sandbox for executing the code
    safe_globals = {"__builtins__": {"print": print, "math": math}}  # Restrict built-ins
    safe_locals = {}

    # Capture the output
    output_buffer = io.StringIO()
    result = {"success": False, "output": None, "error": None}

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals, safe_locals)
        result["success"] = True
        result["output"] = output_buffer.getvalue()
    except Exception as e:
        result["error"] = str(e)
    finally:
        output_buffer.close()

    return result


add_tool = FunctionTool.from_defaults(fn=add_integers)
mul_tool = FunctionTool.from_defaults(fn=mul_integers)
div_tool = FunctionTool.from_defaults(fn=div_integers)
search_tool = FunctionTool.from_defaults(fn=web_search)
code_exec_tool = FunctionTool.from_defaults(fn=execute_code)

def return_tool_list():
    return [add_tool, mul_tool, div_tool, search_tool, code_exec_tool]