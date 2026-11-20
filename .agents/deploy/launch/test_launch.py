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

"""Exercise launch commands and checkpoint validation without allocating GPUs."""

import json
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from preflight_model import inspect_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def checkpoint(root, profile):
    root.mkdir()
    quant = dict(
        quant_method="fp8",
        weight_block_size=[32, 32],
        scale_fmt="ue8m0",
        expert_dtype="fp4",
    )
    if profile == "fp4":
        quant.update(
            moe_quant_algo="NVFP4",
            group_size=16,
            quantized_layers={
                f"layers.{i}.ffn.experts": dict(group_size=16, quant_algo="NVFP4")
                for i in range(2)
            },
            engram_embed_quant="mxfp4",
            engram_embed_block_size=32,
            engram_embed_scale_dtype="ue8m0",
            engram_embed_layers=[1],
        )
    (root / "config.json").write_text(
        json.dumps(
            dict(
                model_type="deepseek_v41",
                text_config=dict(num_hidden_layers=2, engram_layer_ids=[1]),
                quantization_config=quant,
            )
        )
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    (root / "encoding").mkdir()
    (root / "encoding/encoding.py").write_text(
        "def encode_messages(messages): return messages\n"
    )
    header = json.dumps(
        {"weight": dict(dtype="F32", shape=[1], data_offsets=[0, 4])}
    ).encode()
    header += b" " * (-len(header) % 8)
    (root / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 4
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    return root


@pytest.fixture
def environment(tmp_path):
    venv = tmp_path / "venv with spaces"
    (venv / "bin").mkdir(parents=True)
    shim = venv / "bin/python"
    shim.write_text(f"#!{sys.executable}\n" + """import json, os, sys
if sys.argv[1:3] == ['-m', 'tokenspeed.runtime.entrypoints.engine']:
    print(json.dumps({'args': sys.argv[1:], 'env': {key: os.environ.get(key) for key in [
        'TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN', 'TOKENSPEED_MOE_ROUTING_SIMULATION',
        'TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING', 'DECODE_BENCH_OBSERVE', 'DECODE_BENCH_RUN', 'PREFILL_BENCH_OBSERVE',
        'TOKENSPEED_SM120_DEEPEP_WIRE', 'TOKENSPEED_SM120_DEEPEP_CAPACITY',
        'TOKENSPEED_SM120_PREFILL_CAKE_ROWS', 'TOKENSPEED_SM120_CAKE_DECODE_MLA_WIDE',
        'TOKENSPEED_SM120_EPLB_FUSED_RECORD']}}))
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
""")
    shim.chmod(0o755)
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/ptxas").write_text("#!/bin/sh\nexit 0\n")
    (cuda / "bin/ptxas").chmod(0o755)
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LD_LIBRARY_PATH")
        if key in os.environ
    }
    env.update(
        VENV=str(venv),
        STATE_DIR=str(tmp_path / "state"),
        CUDA_HOME=str(cuda),
        GATEWAY_HOST="gateway.example",
        NODE_HOST="node.example",
        NODE_RANK="0",
        P_HANDSHAKE_PORT="18510",
        D_HANDSHAKE_PORT="18511",
        P_DIST_INIT_ADDR="prefill.example:19400",
        D_DIST_INIT_ADDR="decode.example:19500",
    )
    return env


def launch(profile, role, env, *args):
    path = ROOT / "deploy" / profile / "launch" / f"{role}.sh"
    return subprocess.run(
        ["bash", str(path), *args], env=env, text=True, capture_output=True, timeout=20
    )


@pytest.mark.parametrize("profile", ["w4a8", "fp4"])
def test_profiles_validate_checkpoint_and_preserve_extra_arguments(
    tmp_path, environment, profile
):
    environment["MODEL_PATH"] = str(checkpoint(tmp_path / "model", profile))
    result = launch(
        profile,
        "prefill",
        environment,
        "--served-model-name",
        "custom model",
        "--seed",
        "123",
    )
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["args"]
    assert command[-4:] == ["--served-model-name", "custom model", "--seed", "123"]
    assert command[command.index("--query-context-parallel-size") + 1] == "8"
    assert ("--moe-mxfp4-fp8-activation" in command) == (profile == "w4a8")
    assert (
        json.loads(
            (Path(environment["STATE_DIR"]) / "setup/model-identity.json").read_text()
        )["profile"]
        == profile
    )


def test_decode_rejects_wrong_rank_and_missing_required_environment(
    tmp_path, environment
):
    environment["MODEL_PATH"] = str(checkpoint(tmp_path / "model", "w4a8"))
    environment["NODE_RANK"] = "2"
    result = launch("w4a8", "decode", environment)
    assert result.returncode != 0
    assert "NODE_RANK must be 0 or 1" in result.stderr
    environment["NODE_RANK"] = "1"
    environment.pop("D_DIST_INIT_ADDR")
    result = launch("w4a8", "decode", environment)
    assert result.returncode != 0
    assert "set D_DIST_INIT_ADDR" in result.stderr


def test_decode_uses_two_nodes_and_clears_performance_simulation(tmp_path, environment):
    environment["MODEL_PATH"] = str(checkpoint(tmp_path / "model", "w4a8"))
    environment["NODE_RANK"] = "1"
    for key in [
        "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN",
        "TOKENSPEED_MOE_ROUTING_SIMULATION",
        "TOKENSPEED_V41_FORCE_PHYSICAL_ROUTING",
        "DECODE_BENCH_OBSERVE",
        "DECODE_BENCH_RUN",
        "PREFILL_BENCH_OBSERVE",
    ]:
        environment[key] = "1"
    result = launch("w4a8", "decode", environment)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    args = observed["args"]
    for flag, expected in [
        ("--nnodes", "2"),
        ("--node-rank", "1"),
        ("--world-size", "16"),
        ("--ep-size", "16"),
        ("--data-parallel-size", "16"),
    ]:
        assert args[args.index(flag) + 1] == expected
    assert observed["env"]["TOKENSPEED_SM120_DEEPEP_WIRE"] == "elastic_fp8_bf16"
    assert observed["env"]["TOKENSPEED_SM120_DEEPEP_CAPACITY"] == "128"
    assert all(
        observed["env"][key] is None
        for key in observed["env"]
        if key
        not in ["TOKENSPEED_SM120_DEEPEP_WIRE", "TOKENSPEED_SM120_DEEPEP_CAPACITY"]
    )


def test_dry_run_prints_parseable_command_and_does_not_launch_engine(
    tmp_path, environment
):
    environment.update(
        MODEL_PATH=str(checkpoint(tmp_path / "model", "fp4")), DRY_RUN="1"
    )
    result = launch("fp4", "decode", environment, "--served-model-name", "a model")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:3] == [
        str(Path(environment["VENV"]) / "bin/python"),
        "-m",
        "tokenspeed.runtime.entrypoints.engine",
    ]
    assert command[-2:] == ["--served-model-name", "a model"]
    assert not Path(environment["STATE_DIR"]).exists()


def test_model_validation_rejects_wrong_precision_missing_shard_and_truncation(
    tmp_path,
):
    model = checkpoint(tmp_path / "model", "fp4")
    assert inspect_checkpoint(model, "fp4")["indexed_tensors"] == 1
    with pytest.raises(ValueError, match="MXFP4 routed experts"):
        inspect_checkpoint(model, "w4a8")
    shard = model / "model-00001-of-00001.safetensors"
    original = shard.read_bytes()
    shard.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="payload size"):
        inspect_checkpoint(model, "fp4")
    shard.unlink()
    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(model, "fp4")


@pytest.mark.parametrize(
    "role,dp,ep,chunk,interval,cake_rows,fused",
    [
        ("prefill", "8", "8", "4224", "1000", "1024", "0"),
        ("decode", "16", "16", "8192", "10000", "0", "1"),
    ],
)
def test_dp8_eplb_profile(tmp_path, environment, role, dp, ep, chunk, interval, cake_rows, fused):
    environment.update(
        MODEL_PATH=str(checkpoint(tmp_path / "model", "w4a8")),
        W4A8_PD_PROFILE="dp8-eplb",
    )
    result = launch("w4a8", role, environment)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    args = observed["args"]
    assert "--enable-eplb" in args
    for flag, expected in (
        ("--tensor-parallel-size", "1"),
        ("--query-context-parallel-size", "1"),
        ("--data-parallel-size", dp),
        ("--ep-size", ep),
        ("--chunked-prefill-size", chunk),
        ("--ep-num-redundant-experts", "16"),
        ("--ep-dispatch-algorithm", "static"),
        ("--eplb-rebalance-num-iterations", interval),
    ):
        assert args[args.index(flag) + 1] == expected
    assert observed["env"]["TOKENSPEED_SM120_PREFILL_CAKE_ROWS"] == cake_rows
    assert observed["env"]["TOKENSPEED_SM120_EPLB_FUSED_RECORD"] == fused
    assert observed["env"]["TOKENSPEED_SM120_CAKE_DECODE_MLA_WIDE"] == (
        "1" if role == "decode" else None
    )
