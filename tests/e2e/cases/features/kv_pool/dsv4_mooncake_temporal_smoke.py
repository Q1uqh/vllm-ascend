# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Check DeepSeek V4 Flash temporal P/D handoff and optionally compare resident generation."""

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
    return tokens, result.get("kv_transfer_params"), elapsed, choice.get("prompt_token_ids")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="D-first layerwise D2D proxy URL")
    parser.add_argument("--prefill", help="Legacy snapshot P URL")
    parser.add_argument("--decode", help="Legacy snapshot D URL")
    parser.add_argument("--resident")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=Path, help="JSON array of token ID arrays")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if not args.proxy and not (args.prefill and args.decode):
        parser.error("provide --proxy for D2D, or both --prefill and --decode for legacy snapshot handoff")
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
        expected = None
        if args.resident:
            expected, _, _, _ = complete(args.resident, args.model, prompt, args.output_tokens, args.timeout)
        for attempt in range(args.repeat):
            if args.proxy:
                actual, _, elapsed, _ = complete(args.proxy, args.model, prompt, args.output_tokens, args.timeout)
                if expected is not None and actual != expected:
                    raise AssertionError(f"Resident/D2D token IDs differ for prompt {index}, attempt {attempt}")
                print(
                    f"DSV4_ACCEPT prompt={index} tokens={len(prompt)} attempt={attempt} "
                    f"transport=d2d elapsed_s={elapsed:.3f} "
                    f"resident_compared={expected is not None} result=pass",
                    flush=True,
                )
                continue
            first, transfer, prefill_s, processed_prompt = complete(args.prefill, args.model, prompt, 1, args.timeout)
            if (
                not processed_prompt
                or not transfer
                or transfer.get(_HANDOFF, {}).get("end_token") != len(processed_prompt)
            ):
                raise RuntimeError(f"Prefill response has no matching snapshot for prompt {index}")
            rest, _, decode_s, _ = complete(
                args.decode,
                args.model,
                processed_prompt + first,
                args.output_tokens - 1,
                args.timeout,
                transfer=transfer,
            )
            if expected is not None and first + rest != expected:
                raise AssertionError(f"Resident/P-D token IDs differ for prompt {index}, attempt {attempt}")
            print(
                f"DSV4_ACCEPT prompt={index} tokens={len(processed_prompt)} attempt={attempt} "
                f"prefill_s={prefill_s:.3f} decode_s={decode_s:.3f} "
                f"resident_compared={expected is not None} result=pass",
                flush=True,
            )


if __name__ == "__main__":
    main()
