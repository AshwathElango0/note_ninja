import contextlib
import io

def execute_code(code: str) -> dict:
    """
    Executes Python code and returns the result or error.
    
    Parameters:
        code (str): The Python code to execute.
    
    Returns:
        dict: A dictionary with 'success', 'output', and 'error' keys.
    """
    # Sandbox for executing the code
    safe_globals = {"__builtins__": {"print": print}}  # Restrict built-ins
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

# Example Usage
code_snippet = """
x = 10
y = 20
print('The sum is:', x + y)
"""

response = execute_code(code_snippet)
if response["success"]:
    print("Output:", response["output"])
else:
    print("Error:", response["error"])
