# SM120 distributed transport tests

`test_sm120_elastic_distributed.py` exercises actual Elastic FP8 dispatch/BF16
combine with strict weighted-identity references (`rtol=atol=0`), zero-row peers,
masked/skew routes, graph replay with changed routes and doubled inputs, and
serial target/draft/target reuse. Its existing parametrization collects 9 cases
per rank. Invoke it with `torchrun` using 2, 4, 8, 16 or 32 CUDA processes across the
intended nodes in a matching freshly built environment. Single-process pytest
skips the suite and does not validate a distributed configuration.

`TOKENSPEED_SM120_DEEPEP_NICS` supplies a requested visible-GPU-to-NIC map:
the adapter sets `EP_NIC_NAME` and `NVSHMEM_HCA_LIST` before buffer construction.
These assignments do not prove that Elastic's NCCL communicator/GIN provider
selected or pinned that HCA. Configure NCCL's HCA allowlist for the actual
network; verify the selected provider and interfaces from that run's NCCL
initialization/network logs. Optional `NCCL_DEBUG=INFO` with
`NCCL_DEBUG_SUBSYS=INIT,NET,ENV` changes logging only; constructor-only
`EP_BUFFER_DEBUG=1` can record buffer setup. Distinguish requested settings from
observed selection, and do not apply a new test's observations retrospectively
to an earlier run.

Two local ranks provide limited numerical coverage. Running the unchanged 9
cases on the intended EP16 hosts extends that coverage without loading a model;
a finite pass does not exclude rare transport/visibility interleavings. Full
model timing, real speculation and dataset quality remain owned by
[the reproducibility recipes](../../../../.agents/README.md).
