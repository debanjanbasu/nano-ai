"""Tool-calling smoke for the micro-ai server: OpenAI `tools`, streaming, tool results, agent flow.

Usage: python smoke_toolcall.py [BASE_URL]   (default http://127.0.0.1:18093/v1)
"""

import json
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:18093/v1"
M = "minicpm-v-4.7-1b"
weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
read_file = {"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
run_shell = {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return its output.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}


def show(title, req, resp):
    print(f"\n### {title}\nREQUEST:", json.dumps(req, ensure_ascii=False))
    print("RESPONSE:", json.dumps(resp, ensure_ascii=False, indent=1))


def post(req):
    r = urllib.request.Request(f"{BASE}/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=300)


def call(title, **req):
    req = {"model": M, "max_tokens": 200, **req}
    d = json.load(post(req))
    d.pop("telemetry", None)
    show(title, req, d)
    return d


def stream(title, **req):
    req = {"model": M, "max_tokens": 200, "stream": True, **req}
    chunks = [json.loads(line[6:]) for raw in post(req)
              if (line := raw.decode().strip()).startswith("data: ") and line != "data: [DONE]"]
    print(f"\n### {title}\nREQUEST:", json.dumps(req, ensure_ascii=False))
    print("CHUNKS:")
    for ch in chunks:
        print(" ", json.dumps({k: ch[k] for k in ("choices", "usage") if k in ch}, ensure_ascii=False))

wq = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What's the weather in Paris?"}]
r = call("(a) get_weather non-streaming", messages=wq, tools=[weather])
stream("(a) get_weather streaming", messages=wq, tools=[weather])

msg = r["choices"][0]["message"]
tc = msg.get("tool_calls")
if tc:
    follow = wq + [{"role": "assistant", "content": msg.get("content"), "tool_calls": tc},
                   {"role": "tool", "tool_call_id": tc[0]["id"], "content": json.dumps({"temperature": 18, "unit": "celsius", "condition": "sunny"})}]
    call("(b) follow-up with tool result", messages=follow, tools=[weather])

agent = [{"role": "system", "content": "You are a coding agent working in a repository. Use tools to inspect files and run commands."},
         {"role": "user", "content": "Show me what is in src/main.rs."}]
call("(c) coding agent read_file/run_shell", messages=agent, tools=[read_file, run_shell])
stream("(c) coding agent streaming", messages=agent, tools=[read_file, run_shell])

call("(d) tool_choice=required", messages=[{"role": "user", "content": "How many files are in the current directory?"}],
     tools=[read_file, run_shell], tool_choice="required")
call("(d) named tool_choice run_shell", messages=[{"role": "user", "content": "Check the git status."}],
     tools=[read_file, run_shell], tool_choice={"type": "function", "function": {"name": "run_shell"}})
call("stop + plain", messages=[{"role": "user", "content": "Count from 1 to 10 separated by spaces."}], stop=["5"])
