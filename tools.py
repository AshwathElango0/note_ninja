from llama_index.core.tools import FunctionTool
from tavily import TavilyClient
import threading

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

def execute_with_timeout(code: str, timeout=2):
    """Executes Python code provided as a string, with a timeout of 2 seconds."""
    result = {}

    def target():
        try:
            exec_globals = {}
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            result["result"] = exec_locals
        except Exception as e:
            result["error"] = str(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return {"error": "Execution timed out"}
    return result

def execute_with_timeout_secure(code, timeout=2):
    """
    Executes Python code with a timeout and security restrictions.
    Restricts access to built-ins and provides a controlled execution environment.
    """
    result = {}

    def target():
        try:
            # Define a restricted set of built-ins
            safe_builtins = {
                "print": print,
                "range": range,
                "len": len,
                "abs": abs,
                "sum": sum,
                "min": min,
                "max": max,
                # Add other safe functions as needed
            }

            # Create restricted global and local namespaces
            exec_globals = {"__builtins__": safe_builtins}
            exec_locals = {}

            # Execute the code
            exec(code, exec_globals, exec_locals)

            # Store the result
            result["result"] = exec_locals
        except Exception as e:
            # Capture and return any errors
            result["error"] = str(e)

    # Create and start a thread for code execution
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    # Check for timeout
    if thread.is_alive():
        return {"error": "Execution timed out"}

    return result


add_tool = FunctionTool.from_defaults(fn=add_integers)
mul_tool = FunctionTool.from_defaults(fn=mul_integers)
div_tool = FunctionTool.from_defaults(fn=div_integers)
search_tool = FunctionTool.from_defaults(fn=web_search)
code_exec_tool = FunctionTool.from_defaults(fn=execute_with_timeout_secure)

def return_tool_list():
    return [add_tool, mul_tool, div_tool, search_tool]