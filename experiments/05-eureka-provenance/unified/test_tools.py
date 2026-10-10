"""Mind v3 tool tests — no model, no GPU.  python unified\\test_tools.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mind_v3 import calc, numbers, tool_call

assert calc("23*17-95") == "296"
assert calc("29995*1823") == "54680885"
assert calc("7*8/2") == "28"
assert calc("100/4") == "25"
assert calc("2**10") == calc("2^10") == "1024"
assert calc("-3 + 5") == "2"
assert calc("7/2") == "3.5"
for bad in ["import os", "__import__('os')", "a+1", "(1).real", "2 if 1 else 3", "1891"]:
    try:
        calc(bad)
        raise AssertionError(bad)
    except (ValueError, KeyError, SyntaxError):
        pass

assert tool_call('<tool_call>\n{"name": "calc", "arguments": {"expression": "1+1"}}\n</tool_call>') == ("calc", {"expression": "1+1"})
assert tool_call('Using calc: {"name": "calc", "arguments": {"expression": "47*8"}} The answer is 376.') == ("calc", {"expression": "47*8"})
assert tool_call("just an answer") is None
assert tool_call("<tool_call>{not json}</tool_call>") is None

assert numbers("The result is 54,680,885 and 12.5") == {"54680885", "12.5"}
print("tool tests: ok")
