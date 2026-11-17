# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Generate one common prefix and unique suffixes for the three Decode cohorts."""

import hashlib
import json
import random


def common_prompt(vocab, context=131072, seed=37):
    return random.Random(seed).choices(vocab, k=context)


def request_tokens(
    prompt, vocab, batch, rank, index, tail_tokens=6656, seed=37, cohort=1
):
    assert cohort in (0, 1, 2)
    key = f"DECODE:{seed}:{cohort}:{batch}:{rank}:{index}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
    return prompt[:-tail_tokens] + rng.choices(vocab, k=tail_tokens)


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
