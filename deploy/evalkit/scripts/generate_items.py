#!/usr/bin/env python3
"""
Authors the entire ~100-item item bank as JSON files under items/<category>/.

This is a one-shot content generator, not something run_eval.py depends on
at eval time -- it exists so item authorship is reviewable as code and so
new items can be added by extending the lists below and re-running. Per the
README, changing the item bank means re-baselining (old results/<tag>/ runs
are no longer comparable to new ones).

code_exec expected_stdout values are computed here by actually *running* a
reference implementation through the exact same scoring.run_python() used
at eval time, so they can't drift from hand-computed arithmetic.

long_ctx expected sentinel numbers are pulled from corpus.build_corpus(),
which is the same deterministic builder run_eval.py uses at request time.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scoring  # noqa: E402
import corpus  # noqa: E402

ITEMS_DIR = ROOT / "items"


def write_item(category, idx, item):
    d = ITEMS_DIR / category
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"item_{idx:03d}.json"
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n")
    return path


# ===========================================================================
# tool_call (20)
# ===========================================================================

TOOLS = {
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    "get_forecast": {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get a multi-day weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "description": "Number of days to forecast"},
                },
                "required": ["city", "days"],
            },
        },
    },
    "get_weather_alerts": {
        "type": "function",
        "function": {
            "name": "get_weather_alerts",
            "description": "Get active severe weather alerts for a region.",
            "parameters": {
                "type": "object",
                "properties": {"region": {"type": "string"}},
                "required": ["region"],
            },
        },
    },
    "calculate": {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    "get_stock_price": {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    "get_time": {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a timezone.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        },
    },
    "send_email": {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    "create_calendar_event": {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["title", "time"],
            },
        },
    },
    "search_web": {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "translate_text": {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text into a target language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {"type": "string"},
                },
                "required": ["text", "target_language"],
            },
        },
    },
    "book_flight": {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    "book_hotel": {
        "type": "function",
        "function": {
            "name": "book_hotel",
            "description": "Book a hotel room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "nights": {"type": "integer"},
                },
                "required": ["city", "nights"],
            },
        },
    },
    "run_snippet": {
        "type": "function",
        "function": {
            "name": "run_snippet",
            "description": "Run a code snippet and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    "lint_code": {
        "type": "function",
        "function": {
            "name": "lint_code",
            "description": "Lint a block of source code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    "format_json": {
        "type": "function",
        "function": {
            "name": "format_json",
            "description": "Pretty-print a JSON payload string.",
            "parameters": {
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"],
            },
        },
    },
    "convert_currency": {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    "get_exchange_rate": {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the exchange rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {"base": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["base", "quote"],
            },
        },
    },
    "search_flights": {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                    "nonstop": {"type": "boolean", "description": "Optional: restrict to nonstop flights"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    "set_reminder": {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "time": {"type": "string"},
                    "repeat": {
                        "type": "string",
                        "enum": ["daily", "weekly", "none"],
                        "description": "Optional recurrence, defaults to none",
                    },
                },
                "required": ["text", "time"],
            },
        },
    },
    "create_ticket": {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Open a support ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Optional priority",
                    },
                },
                "required": ["title", "description"],
            },
        },
    },
    "set_thermostat": {
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "description": "Set the thermostat target temperature.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number"},
                    "unit": {"type": "string", "enum": ["C", "F"]},
                },
                "required": ["temperature", "unit"],
            },
        },
    },
    "assign_ticket": {
        "type": "function",
        "function": {
            "name": "assign_ticket",
            "description": "Assign a ticket to a team.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}, "assignee": {"type": "string"}},
                "required": ["ticket_id", "assignee"],
            },
        },
    },
    "lookup_user": {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "Look up a user by id.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    "get_order_status": {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status of an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    "track_package": {
        "type": "function",
        "function": {
            "name": "track_package",
            "description": "Track a package by tracking number.",
            "parameters": {
                "type": "object",
                "properties": {"tracking_number": {"type": "string"}},
                "required": ["tracking_number"],
            },
        },
    },
    "search_product": {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "Search the catalog for a product.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "add_to_cart": {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add a product to the shopping cart.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}},
                "required": ["product_id", "quantity"],
            },
        },
    },
    "checkout": {
        "type": "function",
        "function": {
            "name": "checkout",
            "description": "Check out the current cart.",
            "parameters": {
                "type": "object",
                "properties": {"cart_total": {"type": "number"}},
                "required": ["cart_total"],
            },
        },
    },
    "convert_temp": {
        "type": "function",
        "function": {
            "name": "convert_temp",
            "description": "Convert a temperature value between units.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string", "enum": ["C", "F"]},
                    "to_unit": {"type": "string", "enum": ["C", "F"]},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    "authenticate_user": {
        "type": "function",
        "function": {
            "name": "authenticate_user",
            "description": "Authenticate a user and return a session token.",
            "parameters": {
                "type": "object",
                "properties": {"username": {"type": "string"}, "password": {"type": "string"}},
                "required": ["username", "password"],
            },
        },
    },
    "get_account_balance": {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Get the account balance for an authenticated session.",
            "parameters": {
                "type": "object",
                "properties": {"token": {"type": "string"}},
                "required": ["token"],
            },
        },
    },
    "lookup_flight": {
        "type": "function",
        "function": {
            "name": "lookup_flight",
            "description": "Look up gate and status info for a flight.",
            "parameters": {
                "type": "object",
                "properties": {"flight_number": {"type": "string"}},
                "required": ["flight_number"],
            },
        },
    },
    "notify_passenger": {
        "type": "function",
        "function": {
            "name": "notify_passenger",
            "description": "Send a notification message to a passenger.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    "get_user_id": {
        "type": "function",
        "function": {
            "name": "get_user_id",
            "description": "Resolve an email address to an internal user id.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"],
            },
        },
    },
    "get_subscription": {
        "type": "function",
        "function": {
            "name": "get_subscription",
            "description": "Get subscription plan info for a user id.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
}


def tc(name, args=None):
    return {"name": name, "args": args or []}


TOOL_CALL_ITEMS = [
    dict(
        description="single tool, basic call",
        tools=["get_weather"],
        messages=[{"role": "user", "content": "What's the current weather in Paris?"}],
        expected_calls=[tc("get_weather", [{"path": "city", "match": "regex", "pattern": "(?i)paris"}])],
    ),
    dict(
        description="single tool, arithmetic expression arg",
        tools=["calculate"],
        messages=[{"role": "user", "content": "What is 47 * 89? Use the calculate tool to work it out."}],
        expected_calls=[tc("calculate", [{"path": "expression", "match": "regex", "pattern": r"47.*89|89.*47"}])],
    ),
    dict(
        description="tool choice among 3 distractors",
        tools=["get_weather", "get_stock_price", "get_time"],
        messages=[{"role": "user", "content": "What's the current stock price of AAPL?"}],
        expected_calls=[tc("get_stock_price", [{"path": "symbol", "match": "exact", "value": "AAPL"}])],
    ),
    dict(
        description="tool choice among 4 distractors",
        tools=["send_email", "create_calendar_event", "search_web", "translate_text"],
        messages=[{"role": "user", "content": "Translate the word 'Bonjour' into English."}],
        expected_calls=[
            tc(
                "translate_text",
                [
                    {"path": "text", "match": "exact", "value": "Bonjour"},
                    {"path": "target_language", "match": "regex", "pattern": "(?i)\\benglish\\b|\\ben\\b"},
                ],
            )
        ],
    ),
    dict(
        description="tool choice among 2 close distractors",
        tools=["book_flight", "book_hotel"],
        messages=[{"role": "user", "content": "I need a hotel room in Tokyo for 3 nights."}],
        expected_calls=[
            tc(
                "book_hotel",
                [
                    {"path": "city", "match": "regex", "pattern": "(?i)tokyo"},
                    {"path": "nights", "match": "numeric", "value": 3},
                ],
            )
        ],
    ),
    dict(
        description="tool choice among 3 similar weather-domain distractors",
        tools=["get_weather", "get_forecast", "get_weather_alerts"],
        messages=[{"role": "user", "content": "Give me the 5-day weather forecast for Berlin."}],
        expected_calls=[
            tc(
                "get_forecast",
                [
                    {"path": "city", "match": "regex", "pattern": "(?i)berlin"},
                    {"path": "days", "match": "numeric", "value": 5},
                ],
            )
        ],
    ),
    dict(
        description="parallel calls: same tool, two cities",
        tools=["get_weather"],
        messages=[
            {
                "role": "user",
                "content": "What's the weather in New York City and in London? Call the tool once for each city, in the same turn.",
            }
        ],
        expected_calls=[
            tc("get_weather", [{"path": "city", "match": "regex", "pattern": "(?i)new york"}]),
            tc("get_weather", [{"path": "city", "match": "regex", "pattern": "(?i)london"}]),
        ],
        mode="any_order",
    ),
    dict(
        description="parallel calls: two different tools",
        tools=["get_weather", "convert_currency"],
        messages=[
            {
                "role": "user",
                "content": "I'm about to travel to Tokyo: tell me the current weather there, and also convert 100 USD to JPY. Do both in this turn.",
            }
        ],
        expected_calls=[
            tc("get_weather", [{"path": "city", "match": "regex", "pattern": "(?i)tokyo"}]),
            tc(
                "convert_currency",
                [
                    {"path": "amount", "match": "numeric", "value": 100},
                    {"path": "from_currency", "match": "regex", "pattern": "(?i)usd"},
                    {"path": "to_currency", "match": "regex", "pattern": "(?i)jpy"},
                ],
            ),
        ],
        mode="any_order",
    ),
    dict(
        description="parallel calls: schedule + notify in one turn",
        tools=["create_calendar_event", "send_email"],
        messages=[
            {
                "role": "user",
                "content": (
                    "Schedule a meeting titled 'Budget Review' at 3pm tomorrow, and email alice@example.com "
                    "to let her know about it. Do both actions in this turn."
                ),
            }
        ],
        expected_calls=[
            tc("create_calendar_event", [{"path": "title", "match": "contains", "value": "Budget Review"}]),
            tc("send_email", [{"path": "to", "match": "exact", "value": "alice@example.com"}]),
        ],
        mode="any_order",
    ),
    dict(
        description="nested/multiline code argument",
        tools=["run_snippet"],
        messages=[
            {
                "role": "user",
                "content": (
                    "Run this snippet using the tool and tell me what it prints:\n\n"
                    "```\n"
                    "def add(a, b):\n"
                    "    return a + b\n"
                    "\n"
                    "print(add(2, 3))\n"
                    "```"
                ),
            }
        ],
        expected_calls=[
            tc(
                "run_snippet",
                [
                    {"path": "code", "match": "regex", "pattern": r"def\s+add\s*\(\s*a\s*,\s*b\s*\)"},
                    {"path": "code", "match": "contains", "value": "return a + b"},
                ],
            )
        ],
    ),
    dict(
        description="nested/multiline code argument, lint a loop",
        tools=["lint_code"],
        messages=[
            {
                "role": "user",
                "content": (
                    "Please lint this code using the tool:\n\n"
                    "```\n"
                    "total = 0\n"
                    "for i in range(10):\n"
                    "    if i % 2 == 0:\n"
                    "        total += i\n"
                    "print(total)\n"
                    "```"
                ),
            }
        ],
        expected_calls=[
            tc(
                "lint_code",
                [
                    {"path": "code", "match": "contains", "value": "for i in range(10):"},
                    {"path": "code", "match": "regex", "pattern": r"if i % 2 == 0:"},
                ],
            )
        ],
    ),
    dict(
        description="nested/multiline argument with nested JSON braces",
        tools=["format_json"],
        messages=[
            {
                "role": "user",
                "content": (
                    'Please pretty-print this JSON payload using the tool: '
                    '{"a": {"b": [1, 2, 3], "c": null}, "d": "x"}'
                ),
            }
        ],
        expected_calls=[
            tc(
                "format_json",
                [
                    {"path": "payload", "match": "regex", "pattern": r'"a"\s*:\s*\{'},
                    {"path": "payload", "match": "contains", "value": "[1, 2, 3]"},
                ],
            )
        ],
    ),
    dict(
        description="unicode argument: accented city name",
        tools=["get_weather"],
        messages=[{"role": "user", "content": "What's the weather like in São Paulo right now?"}],
        expected_calls=[tc("get_weather", [{"path": "city", "match": "contains", "value": "São Paulo"}])],
    ),
    dict(
        description="unicode argument: non-Latin script text",
        tools=["translate_text"],
        messages=[
            {
                "role": "user",
                "content": "Translate the Japanese greeting こんにちは into English.",
            }
        ],
        expected_calls=[
            tc(
                "translate_text",
                [
                    {"path": "text", "match": "exact", "value": "こんにちは"},
                    {"path": "target_language", "match": "regex", "pattern": "(?i)\\benglish\\b|\\ben\\b"},
                ],
            )
        ],
    ),
    dict(
        description="unicode argument: emoji in subject line",
        tools=["send_email"],
        messages=[
            {
                "role": "user",
                "content": (
                    "Send an email to team@example.com with subject "
                    "'\U0001F389 Congrats on the launch!' and body 'Great work everyone.'"
                ),
            }
        ],
        expected_calls=[
            tc(
                "send_email",
                [
                    {"path": "to", "match": "exact", "value": "team@example.com"},
                    {"path": "subject", "match": "contains", "value": "\U0001F389"},
                ],
            )
        ],
    ),
    dict(
        description="optional-param handling: explicit optional bool requested",
        tools=["search_flights"],
        messages=[{"role": "user", "content": "Find flights from SFO to JFK on 2026-09-01, nonstop only please."}],
        expected_calls=[
            tc(
                "search_flights",
                [
                    {"path": "origin", "match": "exact", "value": "SFO"},
                    {"path": "destination", "match": "exact", "value": "JFK"},
                    {"path": "nonstop", "match": "exact", "value": True},
                ],
            )
        ],
    ),
    dict(
        description="optional-param handling: optional enum inferred from phrasing",
        tools=["set_reminder"],
        messages=[{"role": "user", "content": "Remind me every day at 8am to drink water."}],
        expected_calls=[tc("set_reminder", [{"path": "repeat", "match": "exact", "value": "daily"}])],
    ),
    dict(
        description="optional-param handling: optional field correctly left absent",
        tools=["create_ticket"],
        messages=[
            {
                "role": "user",
                "content": (
                    "Open a support ticket titled 'Printer not working' with description "
                    "'The 3rd floor printer is jammed.' I don't have a priority preference, "
                    "so leave that out."
                ),
            }
        ],
        expected_calls=[
            tc(
                "create_ticket",
                [
                    {"path": "title", "match": "contains", "value": "Printer not working"},
                    {"path": "description", "match": "contains", "value": "jammed"},
                    {"path": "priority", "match": "absent"},
                ],
            )
        ],
    ),
    dict(
        description="tool choice among 3 close finance distractors",
        tools=["get_stock_price", "get_exchange_rate", "convert_currency"],
        messages=[{"role": "user", "content": "Convert 250 EUR to USD for me."}],
        expected_calls=[
            tc(
                "convert_currency",
                [
                    {"path": "amount", "match": "numeric", "value": 250},
                    {"path": "from_currency", "match": "regex", "pattern": "(?i)eur"},
                    {"path": "to_currency", "match": "regex", "pattern": "(?i)usd"},
                ],
            )
        ],
    ),
    dict(
        description="numeric + enum argument typing",
        tools=["set_thermostat"],
        messages=[{"role": "user", "content": "Set the thermostat to 21.5 degrees Celsius."}],
        expected_calls=[
            tc(
                "set_thermostat",
                [
                    {"path": "temperature", "match": "numeric", "value": 21.5, "tol": 0.05},
                    {"path": "unit", "match": "exact", "value": "C"},
                ],
            )
        ],
    ),
]


def build_tool_call_items():
    for i, spec in enumerate(TOOL_CALL_ITEMS, start=1):
        item = {
            "id": f"tool_call_{i:03d}",
            "category": "tool_call",
            "description": spec["description"],
            "request": {
                "messages": spec["messages"],
                "tools": [TOOLS[name] for name in spec["tools"]],
                "tool_choice": "auto",
                "max_tokens": 400,
            },
            "check": {
                "type": "tool_call",
                "expected_calls": spec["expected_calls"],
                "mode": spec.get("mode", "any_order" if len(spec["expected_calls"]) > 1 else "in_order"),
            },
        }
        write_item("tool_call", i, item)


# ===========================================================================
# code_exec (25)
# ===========================================================================

CODE_EXEC_TASKS = [
    dict(
        name="is_prime",
        prompt="Write a Python function `is_prime(n)` that returns True if the integer n is prime, else False (0 and 1 are not prime; handle negatives as False).",
        reference="""
def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True
""",
        harness="for n in [-5, 0, 1, 2, 3, 4, 17, 18, 97, 100, 7919]:\n    print(is_prime(n))",
    ),
    dict(
        name="reverse_string",
        prompt="Write a Python function `reverse_string(s)` that returns the reverse of string s.",
        reference="""
def reverse_string(s):
    return s[::-1]
""",
        harness="for s in ['', 'a', 'racecar', 'hello world', '  spaced  ']:\n    print(reverse_string(s))",
    ),
    dict(
        name="is_palindrome",
        prompt="Write a Python function `is_palindrome(s)` that returns True if s is a palindrome ignoring case, spaces, and punctuation (consider only alphanumeric characters), else False. An empty string counts as a palindrome.",
        reference="""
def is_palindrome(s):
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
""",
        harness=(
            "for s in ['', 'a', 'racecar', 'A man a plan a canal Panama', 'hello', "
            "\"No 'x' in Nixon\", '12321', '12345']:\n    print(is_palindrome(s))"
        ),
    ),
    dict(
        name="fibonacci",
        prompt="Write a Python function `fibonacci(n)` (iterative) returning the nth Fibonacci number, 0-indexed, with fibonacci(0)=0 and fibonacci(1)=1.",
        reference="""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
""",
        harness="for n in [0, 1, 2, 3, 10, 20]:\n    print(fibonacci(n))",
    ),
    dict(
        name="fibonacci_recursive",
        prompt="Write a Python function `fibonacci_recursive(n)` using plain recursion (no memoization required) returning the nth Fibonacci number, 0-indexed, with fibonacci_recursive(0)=0 and fibonacci_recursive(1)=1.",
        reference="""
def fibonacci_recursive(n):
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
""",
        harness="for n in [0, 1, 2, 5, 10, 15]:\n    print(fibonacci_recursive(n))",
    ),
    dict(
        name="factorial_recursive",
        prompt="Write a Python function `factorial(n)` using recursion returning n! (factorial(0) = 1).",
        reference="""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
        harness="for n in [0, 1, 5, 10]:\n    print(factorial(n))",
    ),
    dict(
        name="gcd",
        prompt="Write a Python function `gcd(a, b)` returning the greatest common divisor of non-negative integers a and b (gcd(0, n) = n).",
        reference="""
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
""",
        harness="for a, b in [(0, 5), (5, 0), (12, 18), (17, 13), (100, 75), (0, 0)]:\n    print(gcd(a, b))",
    ),
    dict(
        name="is_anagram",
        prompt="Write a Python function `is_anagram(a, b)` returning True if strings a and b are anagrams of each other (ignore case and spaces), else False.",
        reference="""
def is_anagram(a, b):
    norm = lambda s: sorted(c.lower() for c in s if c != ' ')
    return norm(a) == norm(b)
""",
        harness=(
            "for a, b in [('', ''), ('a', 'a'), ('listen', 'silent'), ('Dormitory', 'Dirty Room'), "
            "('hello', 'world'), ('abc', 'ab')]:\n    print(is_anagram(a, b))"
        ),
    ),
    dict(
        name="word_frequency",
        prompt=(
            "Write a Python function `word_frequency(text)` that returns a dict mapping each "
            "lowercased word (split on whitespace, punctuation stripped from word edges) to its "
            "occurrence count."
        ),
        reference="""
import string

def word_frequency(text):
    counts = {}
    for raw in text.split():
        w = raw.strip(string.punctuation).lower()
        if w:
            counts[w] = counts.get(w, 0) + 1
    return counts
""",
        harness=(
            "for text in ['', 'the cat sat on the mat', 'Hello, hello! HELLO world.']:\n"
            "    d = word_frequency(text)\n"
            "    print(sorted(d.items()))"
        ),
    ),
    dict(
        name="set_ops",
        prompt=(
            "Write a Python function `set_ops(a, b)` where a and b are lists; return a tuple of three "
            "sorted lists: (intersection, union, difference) where difference is elements in a but not in b."
        ),
        reference="""
def set_ops(a, b):
    sa, sb = set(a), set(b)
    return (sorted(sa & sb), sorted(sa | sb), sorted(sa - sb))
""",
        harness=(
            "for a, b in [([1,2,3], [2,3,4]), ([], [1,2]), ([1,1,2], [2,2,3]), ([], [])]:\n"
            "    print(set_ops(a, b))"
        ),
    ),
    dict(
        name="dedup",
        prompt="Write a Python function `dedup(lst)` that returns a new list with duplicate elements removed, preserving the first occurrence order.",
        reference="""
def dedup(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
""",
        harness=(
            "for lst in [[], [1], [1,1,1], [3,1,2,1,3,4], ['a','b','a','c']]:\n    print(dedup(lst))"
        ),
    ),
    dict(
        name="flatten",
        prompt="Write a Python function `flatten(lst)` using recursion that flattens an arbitrarily nested list of ints into a single flat list of ints, preserving order.",
        reference="""
def flatten(lst):
    out = []
    for x in lst:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
""",
        harness=(
            "for lst in [[], [1,2,3], [1,[2,3],[4,[5,6]]], [[[]]], [[1],[2,[3,[4]]]]]:\n"
            "    print(flatten(lst))"
        ),
    ),
    dict(
        name="binary_search",
        prompt="Write a Python function `binary_search(lst, target)` where lst is sorted ascending; return the index of target in lst, or -1 if not present.",
        reference="""
def binary_search(lst, target):
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
        harness=(
            "print(binary_search([], 5))\n"
            "print(binary_search([1,3,5,7,9], 7))\n"
            "print(binary_search([1,3,5,7,9], 1))\n"
            "print(binary_search([1,3,5,7,9], 9))\n"
            "print(binary_search([1,3,5,7,9], 4))\n"
            "print(binary_search([2], 2))"
        ),
    ),
    dict(
        name="run_length_encode",
        prompt=(
            "Write a Python function `run_length_encode(s)` that returns the run-length encoding of "
            "string s as a string, e.g. 'aaab' -> 'a3b1' (each run written as char followed by its count, "
            "no separators). An empty string encodes to an empty string."
        ),
        reference="""
def run_length_encode(s):
    if not s:
        return ''
    out = []
    prev = s[0]
    count = 1
    for c in s[1:]:
        if c == prev:
            count += 1
        else:
            out.append(prev + str(count))
            prev = c
            count = 1
    out.append(prev + str(count))
    return ''.join(out)
""",
        harness=(
            "for s in ['', 'a', 'aaab', 'aabbcc', 'abcdef', 'aaaaaaaaaa']:\n"
            "    print(run_length_encode(s))"
        ),
    ),
    dict(
        name="is_balanced",
        prompt=(
            "Write a Python function `is_balanced(s)` that returns True if all brackets in s among "
            "()[]{} are properly opened/closed/nested (ignore all other characters), else False. "
            "An empty string is balanced."
        ),
        reference="""
def is_balanced(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack
""",
        harness=(
            "for s in ['', '()', '([{}])', '(]', '([)]', 'a(b[c]d)e', '(((', 'no brackets here']:\n"
            "    print(is_balanced(s))"
        ),
    ),
    dict(
        name="longest_common_prefix",
        prompt=(
            "Write a Python function `longest_common_prefix(strs)` where strs is a list of strings; "
            "return the longest common prefix string among them (empty string if the list is empty "
            "or there is no common prefix)."
        ),
        reference="""
def longest_common_prefix(strs):
    if not strs:
        return ''
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ''
    return prefix
""",
        harness=(
            "for strs in [[], ['flower','flow','flight'], ['dog','racecar','car'], "
            "['same','same','same'], ['single']]:\n    print(repr(longest_common_prefix(strs)))"
        ),
    ),
    dict(
        name="sum_of_digits",
        prompt="Write a Python function `sum_of_digits(n)` using recursion that returns the sum of the decimal digits of non-negative integer n.",
        reference="""
def sum_of_digits(n):
    if n < 10:
        return n
    return n % 10 + sum_of_digits(n // 10)
""",
        harness="for n in [0, 7, 19, 123, 999999]:\n    print(sum_of_digits(n))",
    ),
    dict(
        name="caesar_cipher",
        prompt=(
            "Write a Python function `caesar_cipher(s, shift)` that shifts each letter in s by "
            "`shift` positions (wrapping within its case, a-z / A-Z), leaving non-letters unchanged."
        ),
        reference="""
def caesar_cipher(s, shift):
    out = []
    for c in s:
        if c.isupper():
            out.append(chr((ord(c) - 65 + shift) % 26 + 65))
        elif c.islower():
            out.append(chr((ord(c) - 97 + shift) % 26 + 97))
        else:
            out.append(c)
    return ''.join(out)
""",
        harness=(
            "for s, shift in [('abc', 1), ('xyz', 3), ('Hello, World!', 5), ('abc', 0), ('abc', 26), "
            "('ABC', -1)]:\n    print(caesar_cipher(s, shift))"
        ),
    ),
    dict(
        name="camel_to_snake",
        prompt=(
            "Write a Python function `camel_to_snake(s)` using the `re` module that converts a "
            "camelCase or PascalCase identifier to snake_case (all lowercase, words separated by "
            "underscores). Example: 'HTTPServerError' style edge cases are not tested; only simple "
            "camelCase/PascalCase words like 'userId' -> 'user_id'."
        ),
        reference="""
import re

def camel_to_snake(s):
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s)
    return s.lower()
""",
        harness=(
            "for s in ['userId', 'FirstName', 'getHTTPResponse'.replace('HTTP','Http'), 'simple', 'A']:\n"
            "    print(camel_to_snake(s))"
        ),
    ),
    dict(
        name="extract_emails",
        prompt=(
            "Write a Python function `extract_emails(text)` using the `re` module that returns a list "
            "of all email addresses found in text, in the order they appear."
        ),
        reference="""
import re

def extract_emails(text):
    return re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', text)
""",
        harness=(
            "print(extract_emails('contact us at help@example.com or sales@example.co.uk today'))\n"
            "print(extract_emails('no emails here'))\n"
            "print(extract_emails('a@b.com, c.d+tag@e-f.org; end'))\n"
            "print(extract_emails(''))"
        ),
    ),
    dict(
        name="extract_digits",
        prompt=(
            "Write a Python function `extract_digits(text)` using the `re` module that returns a single "
            "string containing only the digit characters found in text, in order (no separators)."
        ),
        reference="""
import re

def extract_digits(text):
    return ''.join(re.findall(r'\\d', text))
""",
        harness=(
            "print(extract_digits('Room 42, Floor 7'))\n"
            "print(extract_digits('no digits'))\n"
            "print(extract_digits('a1b2c3'))\n"
            "print(extract_digits(''))"
        ),
    ),
    dict(
        name="most_common_element",
        prompt=(
            "Write a Python function `most_common_element(lst)` that returns the most frequently "
            "occurring element in list lst, breaking ties by first occurrence in the list. "
            "Return None for an empty list."
        ),
        reference="""
def most_common_element(lst):
    if not lst:
        return None
    counts = {}
    for x in lst:
        counts[x] = counts.get(x, 0) + 1
    best = None
    best_count = -1
    for x in lst:
        if counts[x] > best_count:
            best = x
            best_count = counts[x]
    return best
""",
        harness=(
            "for lst in [[], [1], [1,2,2,3], [3,3,1,1,2], ['a','b','a','b','a']]:\n"
            "    print(most_common_element(lst))"
        ),
    ),
    dict(
        name="chunk_list",
        prompt=(
            "Write a Python function `chunk_list(lst, n)` that splits lst into consecutive chunks of "
            "size n (a list of lists); the final chunk may be shorter than n. Return an empty list if "
            "lst is empty."
        ),
        reference="""
def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]
""",
        harness=(
            "print(chunk_list([], 3))\n"
            "print(chunk_list([1,2,3,4,5], 2))\n"
            "print(chunk_list([1,2,3], 5))\n"
            "print(chunk_list(list(range(9)), 3))"
        ),
    ),
    dict(
        name="matrix_transpose",
        prompt=(
            "Write a Python function `matrix_transpose(matrix)` where matrix is a list of equal-length "
            "row lists; return its transpose as a list of lists. Return an empty list for an empty matrix."
        ),
        reference="""
def matrix_transpose(matrix):
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]
""",
        harness=(
            "print(matrix_transpose([]))\n"
            "print(matrix_transpose([[1,2,3]]))\n"
            "print(matrix_transpose([[1,2],[3,4],[5,6]]))\n"
            "print(matrix_transpose([[1]]))"
        ),
    ),
    dict(
        name="valid_ip",
        prompt=(
            "Write a Python function `valid_ip(s)` using the `re` module that returns True if s is a "
            "syntactically valid IPv4 address (four dot-separated decimal octets, each 0-255, no leading "
            "zeros other than the single digit '0'), else False."
        ),
        reference="""
import re

_OCTET = r'(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])'
_IP_RE = re.compile(r'^' + _OCTET + r'\\.' + _OCTET + r'\\.' + _OCTET + r'\\.' + _OCTET + r'$')

def valid_ip(s):
    return bool(_IP_RE.match(s))
""",
        harness=(
            "for s in ['255.255.255.255', '0.0.0.0', '192.168.1.1', '256.1.1.1', '1.2.3', "
            "'01.1.1.1', '1.2.3.4.5', '']:\n    print(valid_ip(s))"
        ),
    ),
]

assert len(CODE_EXEC_TASKS) == 25, len(CODE_EXEC_TASKS)


def build_code_exec_items():
    # HOLE FIX: this used to hand the model the exact test-driver code
    # (the "harness") verbatim, with the literal concrete inputs baked in,
    # and asked the model to reproduce it. Since scoring only ever checked
    # stdout equality, a "solution" could completely ignore the general
    # problem and just hardcode a lookup table for the visible inputs
    # (`def is_prime(n): return {-5:False,...,7919:True}[n]`) and pass with
    # a perfect score despite implementing nothing. The harness is no
    # longer shown to the model at all -- it's appended by the scorer
    # (scoring.score_code_exec) at execution time, straight from
    # check["harness"]. The model only ever sees the function signature and
    # spec, so it has to implement the real thing to pass on inputs it
    # never saw.
    for i, task in enumerate(CODE_EXEC_TASKS, start=1):
        full_reference = task["reference"].strip("\n") + "\n\n" + task["harness"]
        run = scoring.run_python(full_reference, timeout_sec=10)
        if run["timed_out"] or run["returncode"] != 0:
            raise RuntimeError(f"reference impl for {task['name']} failed: {run}")
        expected_stdout = run["stdout"]

        user_prompt = (
            f"{task['prompt']}\n\n"
            "Reply with a single Python code block containing only your function definition(s) "
            "(imports allowed if needed). Do not include any example usage, test calls, or print "
            "statements of your own -- a separate test harness will be run against your function "
            "afterward. Do not print anything else outside the code block."
        )

        item = {
            "id": f"code_exec_{i:03d}",
            "category": "code_exec",
            "description": task["name"],
            "request": {
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": 900,
            },
            "check": {
                "type": "code_exec",
                "expected_stdout": expected_stdout,
                "timeout_sec": 10,
                "harness": task["harness"],
            },
        }
        write_item("code_exec", i, item)


# ===========================================================================
# long_ctx (15) -- 6 single_fact + 6 multi_fact + 3 purpose
# ===========================================================================

LONG_CTX_SIZES = [8000, 16000, 32000, 64000, 96000, 128000]
PURPOSE_SIZES = [8000, 32000, 128000]


def build_long_ctx_items():
    idx = 1

    for size in LONG_CTX_SIZES:
        _text, meta = corpus.build_corpus(size)
        num50 = meta["sentinels"]["50"]["number"]
        needle = f"SENTINEL-{size}-50"
        item = {
            "id": f"long_ctx_{idx:03d}",
            "category": "long_ctx",
            "description": f"single-fact retrieval, ~{size} target tokens, depth 50%",
            "corpus_tokens": size,
            "request": {
                "question": (
                    f"Search the source code corpus above for a comment line containing the exact text "
                    f"'{needle}'. That line states a magic number. Reply with ONLY that magic number "
                    f"(digits only), nothing else."
                ),
                "max_tokens": 60,
            },
            "check": {"type": "single_fact", "expected_number": num50},
        }
        write_item("long_ctx", idx, item)
        idx += 1

    for size in LONG_CTX_SIZES:
        _text, meta = corpus.build_corpus(size)
        nums = [meta["sentinels"][d]["number"] for d in ("20", "50", "80")]
        needle_prefix = f"SENTINEL-{size}-"
        item = {
            "id": f"long_ctx_{idx:03d}",
            "category": "long_ctx",
            "description": f"multi-fact retrieval, ~{size} target tokens, depths 20/50/80%",
            "corpus_tokens": size,
            "request": {
                "question": (
                    f"Search the source code corpus above for all comment lines containing the text "
                    f"'{needle_prefix}' (there are exactly three: for depths 20, 50, and 80). Each states "
                    f"a magic number. Reply with exactly three lines, one magic number per line, in the "
                    f"order: the -20 one first, then -50, then -80. Digits only, nothing else on each line."
                ),
                "max_tokens": 120,
            },
            "check": {"type": "multi_fact", "expected_numbers": nums},
        }
        write_item("long_ctx", idx, item)
        idx += 1

    for size in PURPOSE_SIZES:
        corpus.build_corpus(size)  # ensure cached
        item = {
            "id": f"long_ctx_{idx:03d}",
            "category": "long_ctx",
            "description": f"purpose summarization, ~{size} target tokens",
            "corpus_tokens": size,
            "request": {
                "question": "In exactly one sentence, state the overall purpose of this codebase.",
                "max_tokens": 100,
            },
            "check": {"type": "purpose", "pattern": r"vllm|inference|LLM|serving"},
        }
        write_item("long_ctx", idx, item)
        idx += 1


# ===========================================================================
# agentic_chain (10)
# ===========================================================================

def step_tool(expect_tool, arg_checks, inject_result, max_tokens=300):
    return {
        "expect_tool": expect_tool,
        "arg_checks": arg_checks,
        "inject_result": inject_result,
        "max_tokens": max_tokens,
    }


# HOLE FIX: a bare "contains one of these words" pattern can't distinguish
# a genuine completion from a failure narrative that happens to use the
# same vocabulary, e.g. "I was NOT able to confirm the email was sent"
# still contains "confirm" and "sent". Every final step gets this negation
# guard by default (paired with scoring.evaluate_final_step); pass
# must_not_pattern=None to disable it for an item where it isn't needed.
_DEFAULT_MUST_NOT_PATTERN = r"\b(not|n't|unable|fail(ed|s)?|couldn't|cannot|can't|never|didn't|wasn't|isn't|error|sorry|unfortunately)\b"


def step_final(pattern, max_tokens=150, must_not_pattern=_DEFAULT_MUST_NOT_PATTERN):
    answer_check = {"pattern": pattern}
    if must_not_pattern:
        answer_check["must_not_pattern"] = must_not_pattern
    return {"final": True, "answer_check": answer_check, "max_tokens": max_tokens}


AGENTIC_CHAIN_ITEMS = [
    dict(
        description="lookup user then email them using the retrieved address",
        tools=["lookup_user", "send_email"],
        user="Look up user u_1042 and send them a welcome email letting them know their account is ready.",
        steps=[
            step_tool(
                "lookup_user",
                [{"path": "user_id", "match": "exact", "value": "u_1042"}],
                {"user_id": "u_1042", "name": "Dana Kim", "email": "dana.kim@example.com"},
            ),
            step_tool(
                "send_email",
                [{"path": "to", "match": "exact", "value_from_previous_result": "email"}],
                {"status": "sent", "message_id": "m_001"},
            ),
            step_final(r"sent|success|done|confirm"),
        ],
    ),
    dict(
        description="order status -> tracking lookup, answer must use tracking result",
        tools=["get_order_status", "track_package"],
        user="What's the delivery ETA for order ORD-9001?",
        steps=[
            step_tool(
                "get_order_status",
                [{"path": "order_id", "match": "exact", "value": "ORD-9001"}],
                {"status": "shipped", "tracking_number": "1Z999AA10123456784"},
            ),
            step_tool(
                "track_package",
                [{"path": "tracking_number", "match": "exact", "value_from_previous_result": "tracking_number"}],
                {"eta": "2026-08-20", "carrier": "UPS"},
            ),
            step_final(r"2026-08-20"),
        ],
    ),
    dict(
        description="3-step: search product -> add to cart -> checkout, answer states order id",
        tools=["search_product", "add_to_cart", "checkout"],
        user="Find a wireless mouse, add 2 to my cart, and check out.",
        steps=[
            step_tool(
                "search_product",
                [{"path": "query", "match": "regex", "pattern": "(?i)wireless mouse|mouse"}],
                {"product_id": "p_77", "price": 49.99},
            ),
            step_tool(
                "add_to_cart",
                [
                    {"path": "product_id", "match": "exact", "value_from_previous_result": "product_id"},
                    {"path": "quantity", "match": "numeric", "value": 2},
                ],
                {"cart_total": 99.98},
            ),
            step_tool(
                "checkout",
                [{"path": "cart_total", "match": "numeric", "value_from_previous_result": "cart_total", "tol": 0.01}],
                {"order_id": "ORD-3301"},
            ),
            step_final(r"ORD-3301"),
        ],
    ),
    dict(
        description="weather -> unit conversion using retrieved temperature",
        tools=["get_weather", "convert_temp"],
        user="What's the temperature in Chicago right now, in Celsius?",
        steps=[
            step_tool(
                "get_weather",
                [{"path": "city", "match": "regex", "pattern": "(?i)chicago"}],
                {"temp_f": 41, "condition": "windy"},
            ),
            step_tool(
                "convert_temp",
                [
                    {"path": "value", "match": "numeric", "value_from_previous_result": "temp_f"},
                    {"path": "to_unit", "match": "exact", "value": "C"},
                ],
                {"result_c": 5.0},
            ),
            step_final(r"5(\.0)?\s*(°|deg)?\s*C"),
        ],
    ),
    dict(
        description="authenticate then fetch balance with returned token",
        tools=["authenticate_user", "get_account_balance"],
        user="Log me in as username 'jsmith' password 'hunter2' and tell me my account balance.",
        steps=[
            step_tool(
                "authenticate_user",
                [
                    {"path": "username", "match": "exact", "value": "jsmith"},
                    {"path": "password", "match": "exact", "value": "hunter2"},
                ],
                {"token": "tok_abc123"},
            ),
            step_tool(
                "get_account_balance",
                [{"path": "token", "match": "exact", "value_from_previous_result": "token"}],
                {"balance": 532.10},
            ),
            step_final(r"532\.1\b|532\.10"),
        ],
    ),
    dict(
        description="create ticket then assign it, answer references the returned ticket id",
        tools=["create_ticket", "assign_ticket"],
        user="Open a ticket titled 'VPN down' and assign it to the support team.",
        steps=[
            step_tool(
                "create_ticket",
                [{"path": "title", "match": "contains", "value": "VPN down"}],
                {"ticket_id": "T-4521"},
            ),
            step_tool(
                "assign_ticket",
                [{"path": "ticket_id", "match": "exact", "value_from_previous_result": "ticket_id"}],
                {"status": "assigned"},
            ),
            step_final(r"T-4521"),
        ],
    ),
    dict(
        description="flight gate lookup then notify passenger with gate info from result",
        tools=["lookup_flight", "notify_passenger"],
        user="Look up flight UA123's gate and let the passenger know.",
        steps=[
            step_tool(
                "lookup_flight",
                [{"path": "flight_number", "match": "exact", "value": "UA123"}],
                {"gate": "B12", "status": "on time"},
            ),
            step_tool(
                "notify_passenger",
                [{"path": "message", "match": "contains", "value_from_previous_result": "gate"}],
                {"status": "sent"},
            ),
            step_final(r"sent|notified|let them know|done"),
        ],
    ),
    dict(
        description="exchange rate lookup then currency conversion using retrieved rate",
        tools=["get_exchange_rate", "convert_currency"],
        user="What is 500 USD worth in GBP right now?",
        steps=[
            step_tool(
                "get_exchange_rate",
                [
                    {"path": "base", "match": "regex", "pattern": "(?i)usd"},
                    {"path": "quote", "match": "regex", "pattern": "(?i)gbp"},
                ],
                {"rate": 0.79},
            ),
            step_tool(
                "convert_currency",
                [{"path": "amount", "match": "numeric", "value": 500}],
                {"converted": 395.0},
            ),
            step_final(r"395(\.0)?"),
        ],
    ),
    dict(
        description="3-step: resolve email to user id -> subscription lookup, answer states plan and renewal date",
        tools=["get_user_id", "get_subscription"],
        user="Look up the subscription plan for the account with email pat@example.com.",
        steps=[
            step_tool(
                "get_user_id",
                [{"path": "email", "match": "exact", "value": "pat@example.com"}],
                {"user_id": "u_500"},
            ),
            step_tool(
                "get_subscription",
                [{"path": "user_id", "match": "exact", "value_from_previous_result": "user_id"}],
                {"plan": "pro", "renews_on": "2026-09-01"},
            ),
            step_final(r"(?=.*pro)(?=.*2026-09-01)"),
        ],
    ),
    dict(
        description="3-step: search flights -> book flight, answer states confirmation code",
        tools=["search_flights", "book_flight"],
        user="Find a flight from SFO to ORD on 2026-09-10 and book it.",
        steps=[
            step_tool(
                "search_flights",
                [
                    {"path": "origin", "match": "exact", "value": "SFO"},
                    {"path": "destination", "match": "exact", "value": "ORD"},
                ],
                {"flight_id": "FL2200", "price": 312.50},
            ),
            step_tool(
                "book_flight",
                [{"path": "origin", "match": "exact", "value": "SFO"}],
                {"confirmation": "CNF-9981"},
            ),
            step_final(r"CNF-9981"),
        ],
    ),
]

assert len(AGENTIC_CHAIN_ITEMS) == 10, len(AGENTIC_CHAIN_ITEMS)


def build_agentic_chain_items():
    for i, spec in enumerate(AGENTIC_CHAIN_ITEMS, start=1):
        item = {
            "id": f"agentic_chain_{i:03d}",
            "category": "agentic_chain",
            "description": spec["description"],
            "tools": [TOOLS[name] for name in spec["tools"]],
            "user": spec["user"],
            "steps": spec["steps"],
        }
        write_item("agentic_chain", i, item)


# ===========================================================================
# json_schema (10)
# ===========================================================================

JSON_SCHEMA_ITEMS = [
    dict(
        prompt=(
            "Return a JSON object describing a person with exactly these fields: "
            "name (non-empty string), age (integer between 0 and 120), "
            "email (string containing an '@'), tags (array of at least 1 string)."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "age", "email", "tags"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
                "email": {"type": "string", "pattern": "@"},
                "tags": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a product with exactly these fields: "
            "sku (string matching the pattern two uppercase letters followed by four digits, e.g. 'AB1234'), "
            "price (number >= 0), in_stock (boolean), categories (array of 1 to 5 strings)."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["sku", "price", "in_stock", "categories"],
            "properties": {
                "sku": {"type": "string", "pattern": r"^[A-Z]{2}\d{4}$"},
                "price": {"type": "number", "minimum": 0},
                "in_stock": {"type": "boolean"},
                "categories": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}},
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a mailing address with exactly these fields: "
            "street (string), city (string), zip (a 5-digit string), "
            "country (one of exactly: 'US', 'CA', 'MX')."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["street", "city", "zip", "country"],
            "properties": {
                "street": {"type": "string", "minLength": 1},
                "city": {"type": "string", "minLength": 1},
                "zip": {"type": "string", "pattern": r"^\d{5}$"},
                "country": {"type": "string", "enum": ["US", "CA", "MX"]},
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a calendar event with exactly these fields: "
            "title (non-empty string), start_time (string in 'YYYY-MM-DD HH:MM' format), "
            "attendees (array of at least 1 string), is_recurring (boolean)."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "start_time", "attendees", "is_recurring"],
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "start_time": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"},
                "attendees": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "is_recurring": {"type": "boolean"},
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a server config with exactly these fields: "
            "hostname (non-empty string), port (integer between 1 and 65535), "
            "tags (array of strings), env (one of exactly: 'dev', 'staging', 'prod')."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["hostname", "port", "tags", "env"],
            "properties": {
                "hostname": {"type": "string", "minLength": 1},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "tags": {"type": "array", "items": {"type": "string"}},
                "env": {"type": "string", "enum": ["dev", "staging", "prod"]},
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a book review with exactly these fields: "
            "title (non-empty string), rating (integer 1 to 5), recommend (boolean), "
            "pros (array of at least 1 string), cons (array of strings, may be empty)."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "rating", "recommend", "pros", "cons"],
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                "recommend": {"type": "boolean"},
                "pros": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "cons": {"type": "array", "items": {"type": "string"}},
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a weather report with exactly these fields: "
            "city (non-empty string), temp_c (number), "
            "condition (one of exactly: 'sunny', 'cloudy', 'rainy', 'snowy', 'windy'), "
            "humidity_pct (integer 0 to 100)."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["city", "temp_c", "condition", "humidity_pct"],
            "properties": {
                "city": {"type": "string", "minLength": 1},
                "temp_c": {"type": "number"},
                "condition": {"type": "string", "enum": ["sunny", "cloudy", "rainy", "snowy", "windy"]},
                "humidity_pct": {"type": "integer", "minimum": 0, "maximum": 100},
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a recipe with exactly these fields: "
            "name (non-empty string), servings (integer >= 1), "
            "ingredients (array of at least 2 strings), minutes (integer >= 1)."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "servings", "ingredients", "minutes"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "servings": {"type": "integer", "minimum": 1},
                "ingredients": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                "minutes": {"type": "integer", "minimum": 1},
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a user profile with exactly these fields: "
            "username (3-16 chars, only lowercase letters/digits/underscore), "
            "email (string containing '@'), active (boolean), "
            "roles (array of at least 1 string, each one of exactly: 'admin', 'user', 'guest')."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["username", "email", "active", "roles"],
            "properties": {
                "username": {"type": "string", "pattern": r"^[a-z0-9_]{3,16}$"},
                "email": {"type": "string", "pattern": "@"},
                "active": {"type": "boolean"},
                "roles": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": ["admin", "user", "guest"]},
                },
            },
        },
    ),
    dict(
        prompt=(
            "Return a JSON object describing a task list with exactly these fields: "
            "title (non-empty string), tasks (array of at least 1 object, each with fields "
            "name (string) and done (boolean))."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "tasks"],
            "properties": {
                "title": {"type": "string", "minLength": 1},
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "done"],
                        "properties": {"name": {"type": "string"}, "done": {"type": "boolean"}},
                    },
                },
            },
        },
    ),
]

assert len(JSON_SCHEMA_ITEMS) == 10, len(JSON_SCHEMA_ITEMS)


def build_json_schema_items():
    for i, spec in enumerate(JSON_SCHEMA_ITEMS, start=1):
        item = {
            "id": f"json_schema_{i:03d}",
            "category": "json_schema",
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": spec["prompt"] + " Respond with ONLY the JSON object, no other text.",
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 400,
            },
            "check": {"type": "json_schema", "schema": spec["schema"]},
        }
        write_item("json_schema", i, item)


# ===========================================================================
# instruction (10)
# ===========================================================================

INSTRUCTION_ITEMS = [
    dict(
        prompt=(
            "Write a short 3-line note about the ocean, one short sentence per line. "
            "Output exactly 3 lines and nothing else (no title, no blank lines)."
        ),
        check={"type": "line_count", "value": 3},
    ),
    dict(
        prompt=(
            "Respond with a short phrase of 5 to 15 characters that contains NO vowels at all "
            "(a, e, i, o, u -- uppercase or lowercase forms all count). Output only the phrase, "
            "nothing else."
        ),
        check={"type": "no_vowels"},
    ),
    dict(
        prompt=(
            "Write a two-sentence product description for a ceramic coffee mug. "
            "Your response must start with the exact word 'INTRODUCING' (all caps)."
        ),
        check={"type": "starts_with", "value": "INTRODUCING"},
    ),
    dict(
        prompt=(
            "Describe your favorite season of the year in exactly 12 words. "
            "Output only those 12 words as a single sentence, nothing else."
        ),
        check={"type": "word_count", "value": 12},
    ),
    dict(
        prompt=(
            "Write a short warning sign message about a wet floor, entirely in uppercase letters. "
            "There must be no lowercase letters anywhere in your response."
        ),
        check={"type": "all_upper"},
    ),
    dict(
        prompt=(
            "Reply with exactly the following 10 characters and nothing else: the digits 0 through 9 "
            "in increasing order, with no spaces or separators between them."
        ),
        check={"type": "exact", "value": "0123456789"},
    ),
    dict(
        prompt=(
            "Write one short sentence about dogs that does NOT contain the letter 'e' anywhere "
            "(case-insensitive). Output only that sentence."
        ),
        check={"type": "no_letter", "value": "e"},
    ),
    dict(
        prompt=(
            "Respond with a syntactically valid-looking IPv4 address (four dot-separated numbers, "
            "each 0-255) and absolutely nothing else -- no words, no punctuation besides the dots."
        ),
        check={
            "type": "regex_fullmatch",
            "pattern": r"(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}",
        },
    ),
    dict(
        prompt=(
            "Is 2 + 2 equal to 4? Reply with exactly one word, either 'YES' or 'NO' in uppercase, "
            "and absolutely nothing else."
        ),
        check={"type": "regex_fullmatch", "pattern": r"YES"},
    ),
    dict(
        prompt=(
            "List exactly 5 programming languages, one per line, each line formatted as "
            "'- <name>' (a hyphen, one space, then the language name). Output exactly 5 lines and "
            "nothing else -- no numbering, no blank lines, no extra commentary."
        ),
        check={
            "type": "regex_fullmatch",
            "pattern": r"-\s+\S.*(?:\n-\s+\S.*){4}",
        },
    ),
]

assert len(INSTRUCTION_ITEMS) == 10, len(INSTRUCTION_ITEMS)


def build_instruction_items():
    for i, spec in enumerate(INSTRUCTION_ITEMS, start=1):
        item = {
            "id": f"instruction_{i:03d}",
            "category": "instruction",
            "request": {
                "messages": [{"role": "user", "content": spec["prompt"]}],
                "max_tokens": 300,
            },
            "check": spec["check"],
        }
        write_item("instruction", i, item)


# ===========================================================================
# loop_probe (10)
# ===========================================================================

LOOP_PROBE_TOPICS = [
    ("the integers from 1 to 100, in order (the item text on each line is just the number itself)", 2000),
    ("distinct English words related to space and astronomy", 2500),
    ("distinct country names (real countries of the world)", 2500),
    ("distinct animal names", 2500),
    ("distinct fruit or vegetable names", 2500),
    ("distinct color names (simple or compound, e.g. 'sky blue' is fine)", 2500),
    ("distinct programming languages, frameworks, or developer tools", 3000),
    ("distinct chemical element names", 2500),
    ("distinct job titles / occupations", 2500),
    ("distinct common English verbs", 2500),
]

assert len(LOOP_PROBE_TOPICS) == 10, len(LOOP_PROBE_TOPICS)


def build_loop_probe_items():
    for i, (topic, max_tokens) in enumerate(LOOP_PROBE_TOPICS, start=1):
        prompt = (
            f"List exactly 100 {topic}. Format each line as '<N>. <item>' with N running from 1 to 100 "
            "in strictly increasing order, one item per line. All 100 items must be distinct "
            "(case-insensitive) -- do not repeat any item. Output only the 100 numbered lines: no title, "
            "no intro, no closing remarks, no blank lines."
        )
        item = {
            "id": f"loop_probe_{i:03d}",
            "category": "loop_probe",
            "description": f"100 distinct: {topic}",
            "request": {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            "check": {"type": "loop_probe", "count": 100},
        }
        write_item("loop_probe", i, item)


# ===========================================================================

def main():
    build_tool_call_items()
    build_code_exec_items()
    build_long_ctx_items()
    build_agentic_chain_items()
    build_json_schema_items()
    build_instruction_items()
    build_loop_probe_items()

    counts = {}
    for cat_dir in sorted((ROOT / "items").iterdir()):
        if cat_dir.is_dir():
            counts[cat_dir.name] = len(list(cat_dir.glob("item_*.json")))
    total = sum(counts.values())
    print("Item counts:")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")
    print(f"  TOTAL: {total}")


if __name__ == "__main__":
    main()
