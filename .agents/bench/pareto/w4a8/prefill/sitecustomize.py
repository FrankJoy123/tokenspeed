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

"""Observe existing host completion waits without adding GPU work."""

import importlib.abc
import importlib.machinery
import os
import sys


class ObserverLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module.ModelExecutor)


class ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "tokenspeed.runtime.execution.model_executor":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = ObserverLoader(spec.loader)
        return spec


def install(ModelExecutor):
    import atexit
    import itertools
    import json
    import logging
    import queue
    import threading
    import time

    from tokenspeed_kernel.compile_monitor import compile_stats

    from tokenspeed.runtime.execution.types import ModelExecutionResult

    pending = {}
    records = queue.SimpleQueue()
    sequence = itertools.count()
    original_execute = ModelExecutor.execute_forward_op
    original_sync = ModelExecutionResult.sync
    logger = logging.getLogger("prefill_benchmark")

    def writer():
        while True:
            record = records.get()
            if record is None:
                return
            logger.info(
                f"PREFILL_BENCHMARK {json.dumps(record, separators=(',', ':'))}"
            )

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()

    def execute(
        self, forward_op, sampling_params_list, dp_metadata=None, *args, **kwargs
    ):
        mapping = self.model_runner.mapping.attn
        sample = None
        if mapping.tp_rank == 0 and mapping.qcp_rank == 0:
            lengths = list(forward_op.input_lengths)
            extends = forward_op.num_extends()
            prefill = sum(lengths[:extends])
            replay = sum(forward_op.extend_replay_lens)
            if not 0 <= replay <= prefill:
                raise ValueError("Invalid Prefill replay geometry")
            sample = {
                "step": next(sequence),
                "rank": mapping.dp_rank,
                "request_ids": list(forward_op.request_ids),
                "num_extends": extends,
                "input_rows": sum(lengths),
                "prefill_rows": prefill,
                "replay_rows": replay,
                "new_rows": prefill - replay,
                "global_rows": (
                    list(dp_metadata.global_num_tokens)
                    if dp_metadata
                    else [sum(lengths)]
                ),
                "global_decodes": (
                    list(dp_metadata.global_num_decodes)
                    if dp_metadata
                    else [len(lengths) - extends]
                ),
                "dispatched_ns": time.perf_counter_ns(),
            }
        result = original_execute(
            self, forward_op, sampling_params_list, dp_metadata, *args, **kwargs
        )
        if sample is not None:
            pending[id(result)] = sample
        return result

    def sync(self):
        original_sync(self)
        completed = time.perf_counter_ns()
        sample = pending.pop(id(self), None)
        if sample is not None:
            compiles = compile_stats()
            sample.update(
                completed_ns=completed,
                serving_compiles=(
                    None if compiles is None else compiles.serving_compiles
                ),
            )
            records.put(sample)

    def shutdown():
        records.put(None)
        thread.join(timeout=3)

    atexit.register(shutdown)
    ModelExecutor.execute_forward_op = execute
    ModelExecutionResult.sync = sync


if os.environ.get("PREFILL_BENCH_OBSERVE") == "1":
    sys.meta_path.insert(0, ObserverFinder())
