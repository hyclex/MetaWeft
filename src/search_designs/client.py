"""Apply the frozen search controls to native chat and embedding requests."""

import base64
from functools import lru_cache
import hashlib
from pathlib import Path

from MetaWeft.service.llm.client import ModelClient
from MetaWeft.src.ace_adapt_reme_task.common import PromptOverflow, SchemaError
from .settings import CAPS, TOKENIZER_HASHES


class OptionalOverflow(PromptOverflow, SchemaError):
    pass


@lru_cache(maxsize=2)
def token_counter(operation):
    name = "tokenizer.json" if operation == "chat" else "cl100k_base.tiktoken"
    path = Path(__file__).with_name(name)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TOKENIZER_HASHES[operation]:
        raise RuntimeError(f"Changed {operation} tokenizer")
    if operation == "chat":
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(path))
        return lambda value: len(tokenizer.encode(value, add_special_tokens=False).ids)
    from tiktoken import Encoding
    from tiktoken_ext.openai_public import cl100k_base
    from types import FunctionType
    ranks = {base64.b64decode(k): int(v) for k, v in (line.split() for line in data.splitlines())}
    factory = FunctionType(cl100k_base.__code__, {**cl100k_base.__globals__,
                           "load_tiktoken_bpe": lambda *args, **kwargs: ranks})
    encoding = Encoding(**factory())
    return lambda value: len(encoding.encode_ordinary(value))


def input_tokens(operation, request):
    count = token_counter(operation)
    if operation == "embedding":
        return sum(map(count, request["input"]))
    return 256 + sum(64 + count(m["role"]) + count(m["content"]) for m in request["messages"])


class SearchClient(ModelClient):
    def __init__(self, *, seed, recorder, **kwargs):
        self.search_seed = seed

        def guarded(operation, request, invoke, **context):
            bound = input_tokens(operation, request)
            output = request.get("max_tokens", 0)
            limit = 65536 if operation == "chat" else 8192
            if bound + output > limit:
                error = RuntimeError if context["phase"] in ("execution", "retrieval") else OptionalOverflow
                raise error(f"{context['phase']} requires {bound} input + {output} output, exceeding {limit}")
            response = recorder(operation, request, invoke, **context)
            usage = getattr(response, "usage", None)
            if (getattr(usage, "prompt_tokens", 0) or 0) > bound or (getattr(usage, "completion_tokens", 0) or 0) > output:
                raise RuntimeError("Reported provider usage exceeded the frozen search bounds")
            return response

        super().__init__(recorder=guarded, **kwargs)

    def chat(self, messages, *, phase, temperature=0.0, **kwargs):
        kwargs.update(seed=self.search_seed, extra_body={"enable_thinking": False},
                      max_tokens=min(kwargs.get("max_tokens", CAPS[phase]), CAPS[phase]))
        return super().chat(messages, phase=phase, temperature=0.0, **kwargs)
