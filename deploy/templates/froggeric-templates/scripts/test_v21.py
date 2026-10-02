import os
import sys
import json
import traceback

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    print("Error: jinja2 is required to run tests. Please install it using 'pip install jinja2'")
    sys.exit(1)

TEMPLATE_FILE = 'chat_template.jinja'
TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    lstrip_blocks=True,
    trim_blocks=True
)

def raise_exception(msg):
    raise Exception(msg)

env.globals['raise_exception'] = raise_exception

try:
    template = env.get_template(TEMPLATE_FILE)
except Exception as e:
    print(f"Error loading template: {e}")
    sys.exit(1)

def run_test(name, messages, tools=None, kwargs=None, expected_in=None, expected_not_in=None, expect_error=False):
    if kwargs is None:
        kwargs = {}
    
    print(f"\n--- Running Test: {name} ---")
    
    try:
        render_kwargs = {'messages': messages, 'add_generation_prompt': True}
        if tools is not None:
            render_kwargs['tools'] = tools
        render_kwargs.update(kwargs)
        
        rendered = template.render(**render_kwargs)
        
        if expect_error:
            print("❌ FAILED: Expected an exception but got none.")
            return False
            
        success = True
        
        if expected_in:
            for ex in expected_in:
                if ex not in rendered:
                    print(f"❌ FAILED: Missing expected string:\n'''{ex}'''")
                    print(f"Rendered:\n{rendered}")
                    success = False
        
        if expected_not_in:
            for n_ex in expected_not_in:
                if n_ex in rendered:
                    print(f"❌ FAILED: Found string that should NOT be present:\n'''{n_ex}'''")
                    print(f"Rendered:\n{rendered}")
                    success = False
                    
        if success:
            print("✅ PASSED")
            return True
        return False
        
    except Exception as e:
        if expect_error:
            print(f"✅ PASSED (Caught expected error: {e})")
            return True
        print(f"❌ FAILED with exception:\n{traceback.format_exc()}")
        return False

# Tests
tests_passed = 0
tests_total = 0

def execute_test(*args, **kwargs):
    global tests_passed, tests_total
    tests_total += 1
    if run_test(*args, **kwargs):
        tests_passed += 1

# 1. auto_disable_thinking_with_tools (Default logic)
execute_test(
    "auto_disable_thinking_with_tools (enabled via kwarg)",
    messages=[{"role": "user", "content": "Hello!"}],
    tools=[{"name": "test_tool"}],
    kwargs={"auto_disable_thinking_with_tools": True},
    expected_in=["<think>\n\n</think>\n\n"], # Should be stripped
)

execute_test(
    "auto_disable_thinking_with_tools (disabled via kwarg -> allows thinking)",
    messages=[{"role": "user", "content": "Hello!"}],
    tools=[{"name": "test_tool"}],
    kwargs={"auto_disable_thinking_with_tools": False},
    expected_in=["<think>\n"],
    expected_not_in=["<think>\n</think>\n"]
)

# 2. inline <|think_on|> overrides auto_disable_thinking_with_tools
execute_test(
    "inline <|think_on|> overrides auto_disable",
    messages=[{"role": "user", "content": "Hello! <|think_on|>"}],
    tools=[{"name": "test_tool"}],
    kwargs={"auto_disable_thinking_with_tools": True},
    expected_in=["<think>\n"],
    expected_not_in=["<think>\n</think>\n", "<|think_on|>"] # Tag must be stripped
)

# 3. Payload truncation
execute_test(
    "max_tool_arg_chars truncation",
    messages=[{"role": "user", "content": "Call tool"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "test", "arguments": {"param": "1234567890"}}}]}],
    kwargs={"max_tool_arg_chars": 5},
    expected_in=["<parameter=param>\n12345\n[TRUNCATED"]
)

execute_test(
    "max_tool_response_chars truncation",
    messages=[{"role": "user", "content": "Do it"}, {"role": "assistant", "content": "calling", "tool_calls": [{"name": "test"}]}, {"role": "tool", "content": "1234567890"}],
    kwargs={"max_tool_response_chars": 5},
    expected_in=["<tool_response>\n12345\n[TRUNCATED"]
)

# 4. Mid-conversation System Prompt
execute_test(
    "mid-conversation system prompt",
    messages=[{"role": "user", "content": "Hello"}, {"role": "system", "content": "Reminder: Be polite"}],
    expected_in=["<|im_start|>system\nReminder: Be polite<|im_end|>"]
)

# 5. Parallel tools delimiter
execute_test(
    "parallel tools delimiter",
    messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "t1"}, {"name": "t2"}]}],
    expected_in=["</function>\n</tool_call>\n\n<tool_call>\n<function=t2>"]
)

# 6. Deep Agent Fallback
execute_test(
    "deep agent fallback (no user message)",
    messages=[{"role": "system", "content": "Sys"}, {"role": "tool", "content": "test"}],
    expected_in=["<|im_start|>system\nSys", "<|im_start|>user\n<tool_response>"]
)

# 7. Error Escalation
execute_test(
    "error escalation warnings",
    messages=[
        {"role": "user", "content": "Do it"}, 
        {"role": "tool", "content": "error: something failed"},
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "error: failed again"}
    ],
    expected_in=["⚠️ SYSTEM WARNING: 2 consecutive tool errors", "<think>\n\n</think>\n\n"]
)

run_test(
    "tool_call_format='json' (override)",
    messages=[{"role": "user", "content": "Call tool"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "test", "arguments": {"par": "1234567890"}}}]}],
    tools=[{"type": "function", "function": {"name": "test", "description": "test tool"}}],
    kwargs={'tool_call_format': 'json'},
    expected_in=[
        'Function calls MUST follow the specified format: a single JSON object with "name" and "arguments"',
        '{"name": "test", "arguments": {"par": "1234567890"}}'
    ]
)

print(f"\n=============================")
print(f"Test Summary: {tests_passed} / {tests_total} passed.")
if tests_passed == tests_total:
    print("All tests passed successfully! 🎉")
    sys.exit(0)
else:
    print("Some tests failed.")
    sys.exit(1)
