"""bfcl CLI with two extra registry entries for MiniCPM-V 4.7 (Qwen3.5 XML tool format) on a micro-ai server.

- minicpm-v-4.7-1b-xml: prompt mode. Renders the official chat template (tools=...) with the HF tokenizer, calls
  /v1/completions, and parses `<tool_call><function=NAME><parameter=P>V</parameter></function></tool_call>`,
  typing each value from the tool's JSON schema as vLLM's qwen3_coder parser does.
- minicpm-v-4.7-1b-FC: function-calling mode. Stock OpenAICompletionsHandler (chat completions with `tools`).
Usage: python bfcl_run.py generate|evaluate ...   (same arguments as `bfcl`)
"""

import ast
import json
import re
import threading
import sys
import types

sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))  # qwen_agent imports unused soundfile at top level

from overrides import override
from bfcl_eval.__main__ import cli
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*(?:</tool_call>)?", re.S)
PARAM_RE = re.compile(r"<parameter[=\s]([^>\n]+)>\n?(.*?)\n?(?:</parameter>|</>)", re.S)
# Tool schemas of the in-flight query, read back when its response is parsed (same thread).
_LOCAL = threading.local()


def typed(raw, schema):
    kind = schema.get("type", "string")
    s = raw.strip()
    if kind == "string":
        en = schema.get("enum")
        if isinstance(en, list) and raw not in en:
            for c in en:
                if isinstance(c, str) and c.lower() == s.lower():
                    return c
        return raw
    try:
        value = json.loads(s)
    except ValueError:
        try:
            value = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            if kind == "integer":
                m = re.match(r"^[+-]?\d+", s)
                if m:
                    return int(m.group(0))
            elif kind == "number":
                m = re.match(r"^[+-]?\d+(?:\.\d+)?", s)
                if m:
                    return float(m.group(0))
            elif kind == "boolean" and s.lower() in ("true", "false", "1", "0", "yes", "no"):
                return s.lower() in ("true", "1", "yes")
            elif kind == "array" and "," in s:
                item_schema = schema.get("items", {})
                return [typed(x.strip().strip("'\""), item_schema) for x in s.strip("[]()").split(",") if x.strip()]
            return raw
    if isinstance(value, tuple):
        value = list(value)
    if kind == "number" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if kind == "array":
        if not isinstance(value, list):
            value = [value]
        item_kind = schema.get("items", {}).get("type")
        if item_kind == "number":
            value = [float(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in value]
        elif item_kind == "integer":
            value = [int(x) if isinstance(x, float) and x.is_integer() else x for x in value]
        elif item_kind == "string":
            value = [str(x) if not isinstance(x, str) else x for x in value]
    return value


class MiniCPMV47XMLHandler(QwenFCHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import transformers
        from pathlib import Path
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        orig = transformers.AutoTokenizer.from_pretrained
        def _load(pretrained_model_name_or_path, *a, **kw):
            path = pretrained_model_name_or_path
            try:
                return orig(path, *a, **kw)
            except Exception:
                env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
                env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
                env.filters["tojson"] = lambda v, indent=None: json.dumps(v, ensure_ascii=False, indent=indent)
                tpl = env.from_string((Path(path) / "chat_template.jinja").read_text())
                return types.SimpleNamespace(
                    model_max_length=262144,
                    tokenize=lambda s: s.split(),
                    apply_chat_template=lambda messages, tools=None, add_generation_prompt=True, tokenize=False: tpl.render(
                        messages=messages, tools=tools, add_generation_prompt=add_generation_prompt
                    ),
                )
        transformers.AutoTokenizer.from_pretrained = _load

    @override
    def _format_prompt(self, messages, function):
        tools = [{"type": "function", "function": f} for f in convert_to_tool(function, GORILLA_TO_OPENAPI, ModelStyle.OSSMODEL)]
        _LOCAL.tools = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
        return self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    @staticmethod
    @override
    def _extract_tool_calls(input_string):
        props = getattr(_LOCAL, "tools", {})
        calls = []
        for name, body in CALL_RE.findall(input_string):
            schema = props.get(name.strip(), {})
            args = {p.strip(): typed(v, schema.get(p.strip(), {})) for p, v in PARAM_RE.findall(body)}
            calls.append({"name": name.strip(), "arguments": args})
        return calls

    @override
    def _parse_query_response_prompting(self, api_response) -> dict:
        data = super()._parse_query_response_prompting(api_response)
        calls = data["model_responses_message_for_chat_history"].get("tool_calls")
        if calls:  # store typed calls (evaluation has no schema) next to the raw text (for debugging)
            data["model_responses"] = json.dumps(
                {"calls": [{c["name"]: c["arguments"]} for c in calls], "raw": api_response.choices[0].text})
        return data

    @staticmethod
    def _decoded(result):
        calls = json.loads(result)["calls"]  # plain text (no tool call) raises here, as BFCL expects
        if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return calls

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return self._decoded(result)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return convert_to_function_call(self._decoded(result))


class MiniCPMV47FCHandler(OpenAICompletionsHandler):
    """Stock FC handler plus a token cap: the OpenAI handler sends no `max_tokens`, so the server
    would generate until its 262K-token context window fills. The OSS handler in the same package
    already caps at 4096."""

    MAX_TOKENS = 4096

    @override
    def _query_FC(self, inference_data: dict):
        message = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        kwargs = {"messages": message, "model": self.model_name, "temperature": self.temperature,
                  "store": False, "max_tokens": self.MAX_TOKENS}
        if len(tools) > 0:
            kwargs["tools"] = tools
        return self.generate_with_backoff(**kwargs)


def config(display, handler):
    return ModelConfig(model_name="minicpm-v-4.7-1b", display_name=display, url="local micro-ai", org="openbmb",
                       license="apache-2.0", model_handler=handler, input_price=None, output_price=None,
                       is_fc_model=True, underscore_to_dot=True)  # both send OpenAI-style tool names (dots -> _)


MODEL_CONFIG_MAPPING["minicpm-v-4.7-1b-xml"] = config("MiniCPM-V-4.7-1B (XML prompt)", MiniCPMV47XMLHandler)
MODEL_CONFIG_MAPPING["minicpm-v-4.7-1b-FC"] = config("MiniCPM-V-4.7-1B (FC)", MiniCPMV47FCHandler)
if __name__ == "__main__":
    cli()
else:  # self-check: `python -c "import bfcl_run"` from .scratch/evals
    _LOCAL.tools = {"f": {"n": {"type": "integer"}, "s": {"type": "string"}, "x": {"type": "number"}, "l": {"type": "array"}}}
    assert MiniCPMV47XMLHandler._extract_tool_calls(
        "ok\n<tool_call>\n<function=f>\n<parameter=n>\n5\n</parameter>\n<parameter=s>\n5\n</parameter>\n"
        "<parameter=x>\n2\n</parameter>\n<parameter=l>\n[1, \"a\"]\n</parameter>\n</function>\n</tool_call>"
    ) == [{"name": "f", "arguments": {"n": 5, "s": "5", "x": 2.0, "l": [1, "a"]}}]
