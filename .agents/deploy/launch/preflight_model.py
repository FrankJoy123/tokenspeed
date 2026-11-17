#!/usr/bin/env python3
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

"""Check local checkpoint metadata and indexed shards before allocating GPUs."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def metadata(root, relative):
    path = root / relative
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Required model file is empty: {relative}")
    return raw


def validate_profile(config, profile):
    quant = config["quantization_config"]
    text = config["text_config"]
    if config["model_type"] != "deepseek_v41":
        raise ValueError("Expected a DeepSeek V4.1 checkpoint")
    if (
        quant.get("quant_method") != "fp8"
        or quant.get("weight_block_size") != [32, 32]
        or quant.get("scale_fmt") != "ue8m0"
    ):
        raise ValueError(
            "Expected original FP8 dense weights with block32 UE8M0 scales"
        )
    if profile == "w4a8":
        if (
            quant.get("expert_dtype") != "fp4"
            or quant.get("moe_quant_algo") is not None
        ):
            raise ValueError("W4A8 requires original MXFP4 routed experts")
        if quant.get("engram_embed_quant", "mxfp8") != "mxfp8":
            raise ValueError("W4A8 recipe requires original MXFP8 Engram tables")
    elif profile == "fp4":
        layers = text["num_hidden_layers"]
        expected = {
            f"layers.{i}.ffn.experts": {"group_size": 16, "quant_algo": "NVFP4"}
            for i in range(layers)
        }
        if (
            quant.get("moe_quant_algo") != "NVFP4"
            or quant.get("group_size") != 16
            or quant.get("quantized_layers") != expected
        ):
            raise ValueError(
                "FP4 recipe requires NVFP4 group16 on exactly the routed experts"
            )
        if (
            quant.get("engram_embed_quant") != "mxfp4"
            or quant.get("engram_embed_block_size") != 32
            or quant.get("engram_embed_scale_dtype") != "ue8m0"
            or quant.get("engram_embed_layers") != text["engram_layer_ids"]
        ):
            raise ValueError(
                "FP4 recipe requires MXFP4 block32 UE8M0 on all Engram layers"
            )
    else:
        raise ValueError("profile must be w4a8 or fp4")


def inspect_checkpoint(model, profile):
    model = model.resolve(strict=True)
    names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "encoding/encoding.py",
        "model.safetensors.index.json",
    ]
    raw = {name: metadata(model, name) for name in names}
    config = json.loads(raw["config.json"])
    validate_profile(config, profile)
    weight_map = json.loads(raw["model.safetensors.index.json"])["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index has no weights")
    shards = {}
    for name in sorted(set(weight_map.values())):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Checkpoint index contains an unsafe shard path")
        path = model / relative
        size = path.stat().st_size
        with path.open("rb") as stream:
            length_raw = stream.read(8)
            if len(length_raw) != 8:
                raise ValueError(f"Truncated safetensors shard: {name}")
            length = struct.unpack("<Q", length_raw)[0]
            if length > size - 8:
                raise ValueError(f"Truncated safetensors header: {name}")
            header_raw = stream.read(length)
        header = json.loads(header_raw)
        tensor_names = set(header) - {"__metadata__"}
        offsets = [header[key]["data_offsets"] for key in tensor_names]
        if (
            not offsets
            or any(
                not isinstance(a, int)
                or not isinstance(b, int)
                or not 0 <= a <= b <= size - 8 - length
                for a, b in offsets
            )
            or max(b for _, b in offsets) != size - 8 - length
        ):
            raise ValueError(
                f"Safetensors payload size disagrees with its header: {name}"
            )
        expected_names = {key for key, shard in weight_map.items() if shard == name}
        if not expected_names.issubset(tensor_names):
            raise ValueError(f"Indexed tensors are absent from shard header: {name}")
        shards[name] = {
            "bytes": size,
            "header_sha256": digest(header_raw),
            "indexed_tensors": len(expected_names),
        }
    return {
        "profile": profile,
        "metadata_sha256": {name: digest(value) for name, value in raw.items()},
        "shards": shards,
        "indexed_tensors": len(weight_map),
        "scope": "Metadata and indexed shard/header completeness; payload bytes are not fully hashed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", choices=["w4a8", "fp4"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_checkpoint(args.model, args.profile)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
