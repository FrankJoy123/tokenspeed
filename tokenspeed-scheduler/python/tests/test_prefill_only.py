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

"""Cache-only scheduling executes every prompt token and never decodes."""

import pytest
from tokenspeed_scheduler import (
    AttnKind,
    CacheGroupConfig,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def scheduler():
    config = SchedulerConfig()
    config.prefix_granularity = 16
    config.max_scheduled_tokens = 32
    config.max_batch_size = 4
    config.num_device_pages = 128
    config.num_snapshot_pages = 1
    config.disable_l2_cache = True
    config.cache_only_prefill = True
    config.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=16,
            total_pages=128,
            attn_kind=AttnKind.Full,
        )
    ]
    return Scheduler(config)


def submit(engine, rid, tokens, outputs=0):
    spec = RequestSpec()
    spec.request_id = rid
    spec.tokens = tokens
    spec.max_new_tokens = outputs
    engine.submit_requests([spec])


def advance(engine, rid, kind):
    event = kind()
    event.request_id = rid
    engine.advance(ExecutionEvent().add_event(event))


def test_prompt_end_waits_for_commit_and_publishes_full_last_page():
    engine = scheduler()
    submit(engine, "prime", list(range(64)))
    first = engine.next_execution_plan().forward[0]
    assert first.input_lengths == [32]
    advance(engine, "prime", ForwardEvent.ExtendResult)
    last = engine.next_execution_plan().forward[0]
    assert last.extend_prefix_lens == [32]
    assert last.input_lengths == [32]
    for _ in range(4):
        assert all(
            not batch.request_ids for batch in engine.next_execution_plan().forward
        )
    advance(engine, "prime", ForwardEvent.ExtendResult)
    advance(engine, "prime", ForwardEvent.Finish)
    engine.next_execution_plan()
    submit(engine, "long", list(range(64)) + list(range(1000, 1032)))
    matched = engine.next_execution_plan().forward[0]
    assert matched.extend_prefix_lens == [64]
    assert matched.input_lengths == [32]


def test_cache_only_refuses_output_tokens():
    with pytest.raises(ValueError, match="max_new_tokens"):
        submit(scheduler(), "bad", [1, 2, 3], outputs=1)
