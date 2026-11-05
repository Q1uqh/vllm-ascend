# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Compare DeepSeek V4 Flash temporal P/D generation to resident generation."""

import argparse
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

_HANDOFF = "ascend_dsv4_snapshot"


def complete(url, model, prompt, count, timeout, transfer=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": count,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "logprobs": 1,
    }
    if transfer is not None:
        payload["kv_transfer_params"] = transfer
    request = Request(
        f"{url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    elapsed = time.monotonic() - start
    choice = result["choices"][0]
    tokens = choice["token_ids"]
    logprobs = choice["logprobs"]["token_logprobs"]
    if len(tokens) != count or len(logprobs) != count or not all(math.isfinite(value) for value in logprobs):
        raise ValueError("Unexpected output length or nonfinite sampled logprobs")
    return tokens, result.get("kv_transfer_params"), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--resident", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=Path, help="JSON array of token ID arrays")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.output_tokens < 2 or args.repeat < 1:
        parser.error("--output-tokens must be at least 2 and --repeat must be positive")
    prompts = json.loads(args.prompt_tokens.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(
            not isinstance(prompt, list) or not prompt or any(type(token) is not int or token < 0 for token in prompt)
            for prompt in prompts
        )
    ):
        parser.error("--prompt-tokens must contain nonempty arrays of nonnegative integer token IDs")

    for index, prompt in enumerate(prompts):
        expected, _, _ = complete(args.resident, args.model, prompt, args.output_tokens, args.timeout)
        for attempt in range(args.repeat):
            first, transfer, prefill_s = complete(args.prefill, args.model, prompt, 1, args.timeout)
            if not transfer or transfer[_HANDOFF]["end_token"] != len(prompt):
                raise RuntimeError(f"Prefill response has no matching snapshot for prompt {index}")
            rest, _, decode_s = complete(
                args.decode,
                args.model,
                prompt + first,
                args.output_tokens - 1,
                args.timeout,
                transfer=transfer,
            )
            if first + rest != expected:
                raise AssertionError(f"Resident/P-D token IDs differ for prompt {index}, attempt {attempt}")
            print(
                f"DSV4_ACCEPT prompt={index} tokens={len(prompt)} attempt={attempt} "
                f"prefill_s={prefill_s:.3f} decode_s={decode_s:.3f} result=pass",
                flush=True,
            )


if __name__ == "__main__":
    main()
