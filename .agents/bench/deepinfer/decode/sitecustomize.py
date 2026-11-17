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

"""Install observation after native host modules have registered their types."""

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

    # Freeze selected behavior and package bytes once at process startup,
    # before model loading/capture; never serialize the complete environment.
    import hashlib
    import itertools
    import json
    import queue
    import threading
    import time
    from pathlib import Path

    import tokenspeed_kernel
    from tokenspeed_kernel.compile_monitor import compile_stats

    import tokenspeed
    from tokenspeed.runtime.execution.types import ModelExecutionResult

    root = Path(os.environ["DECODE_BENCH_RUN"])
    root.mkdir(parents=True, exist_ok=True)
    keys = [
        "TOKENSPEED_EXPERIMENT_SM120_DEEPEP",
        "TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER",
        "TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP",
        "TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT",
        "TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER",
        "TOKENSPEED_ENABLE_SM120_FP4_CANDIDATE_INDEXER",
        "TOKENSPEED_SM120_DEEPEP_WIRE",
        "TOKENSPEED_SM120_DEEPEP_NICS",
        "TOKENSPEED_SM120_DEEPEP_CAPACITY",
        "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN",
        "TOKENSPEED_MOE_ROUTING_SIMULATION",
        "TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING",
        "NCCL_IB_GID_INDEX",
        "NVSHMEM_IB_GID_INDEX",
        "NCCL_SOCKET_IFNAME",
        "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME",
        "NCCL_IB_HCA",
        "NVSHMEM_QP_DEPTH",
        "VIRTUAL_ENV",
    ]
    sources = {
        "runtime_model": Path(tokenspeed.__file__).parent
        / "runtime/models/deepseek_v41.py",
        "indexer": Path(tokenspeed_kernel.__file__).parent
        / "ops/attention/dsv41/triton.py",
    }
    (root / f"process-identity-{os.getpid()}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "starttime": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                "selected_environment": {key: os.environ.get(key) for key in keys},
                "source_sha256": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in sources.items()
                },
            },
            indent=2,
        )
    )

    pending = {}
    generated = {}
    records = queue.SimpleQueue()
    original_execute = ModelExecutor.execute_forward_op
    original_idle = ModelExecutor.execute_idle_forward
    original_sync = ModelExecutionResult.sync
    sequence = itertools.count()

    def writer():
        path = Path(os.environ["DECODE_BENCH_RUN"])
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"steps-{os.getpid()}.jsonl").open("a", buffering=1) as stream:
            while True:
                record = records.get()
                if record is None:
                    return
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()

    def execute(
        self, forward_op, sampling_params_list, dp_metadata=None, *args, **kwargs
    ):
        sample = {
            "step": next(sequence),
            "rank": self.config.global_rank,
            "request_ids": list(forward_op.request_ids),
            "prefill_lengths": list(forward_op.prefill_lengths),
            "num_extends": forward_op.num_extends(),
            "input_rows": sum(forward_op.input_lengths),
            "global_batch": (
                list(dp_metadata.global_batch_size)
                if dp_metadata
                else [len(forward_op.request_ids)]
            ),
            "global_decodes": (
                list(dp_metadata.global_num_decodes)
                if dp_metadata
                else [len(forward_op.request_ids) - forward_op.num_extends()]
            ),
            "all_decode": (
                dp_metadata.all_decode_or_idle
                if dp_metadata
                else forward_op.num_extends() == 0
            ),
            "dispatched_ns": time.perf_counter_ns(),
        }
        if sample["request_ids"] and sample["request_ids"][0].startswith("trace-b"):
            import torch

            label = f"DECODE_TRACE rank={sample['rank']} step={sample['step']} B={len(sample['request_ids'])} rows={sample['input_rows']}"
            with torch.profiler.record_function(label):
                result = original_execute(
                    self, forward_op, sampling_params_list, dp_metadata, *args, **kwargs
                )
        else:
            result = original_execute(
                self, forward_op, sampling_params_list, dp_metadata, *args, **kwargs
            )
        sample.update(
            ep_nic_name=os.environ.get("EP_NIC_NAME"),
            nvshmem_hca_list=os.environ.get("NVSHMEM_HCA_LIST"),
        )
        pending[id(result)] = sample
        return result

    def idle(self, dp_metadata):
        sample = {
            "step": next(sequence),
            "rank": self.config.global_rank,
            "request_ids": [],
            "num_extends": 0,
            "input_rows": 0,
            "global_batch": list(dp_metadata.global_batch_size),
            "global_decodes": list(dp_metadata.global_num_decodes),
            "all_decode": False,
            "idle": True,
        }
        result = original_idle(self, dp_metadata)
        sample.update(
            completed_ns=time.perf_counter_ns(), output_lengths=[], compiles=None
        )
        records.put(sample)
        return result

    def sync(self):
        original_sync(self)
        completed = time.perf_counter_ns()
        sample = pending.pop(id(self), None)
        if sample is not None:
            stats = compile_stats()
            lengths = self.output_lengths.tolist()
            if sample["num_extends"] == 0 and sample["request_ids"]:
                positions = [
                    n + generated.get(rid, 0) - 1
                    for rid, n in zip(sample["request_ids"], sample["prefill_lengths"])
                ]
                sample.update(context_min=min(positions), context_max=max(positions))
            for rid, length in zip(sample["request_ids"], lengths):
                generated[rid] = generated.get(rid, 0) + length
            sample.update(
                completed_ns=completed,
                output_lengths=lengths,
                compiles=None if stats is None else stats.serving_compiles,
            )
            records.put(sample)

    def shutdown():
        records.put(None)
        thread.join(timeout=3)

    atexit.register(shutdown)
    ModelExecutor.execute_forward_op = execute
    ModelExecutor.execute_idle_forward = idle
    ModelExecutionResult.sync = sync


if os.environ.get("DECODE_BENCH_OBSERVE") == "1":
    sys.meta_path.insert(0, ObserverFinder())
