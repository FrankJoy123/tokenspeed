---
name: mmmu-pro-regression
description: 冻结并运行官方 MMMU-Pro 图像子集；保留固定210题制品兼容回归与三轮审计。
---

# 图像回归

在仓库根目录使用独立 Python3.12 CPU venv：

```bash
python3.12 -m venv _local/eval-venv
source _local/eval-venv/bin/activate
python -m pip install -r .agents/eval/mmmu-pro/skill/requirements.txt
```

从[固定revision](https://huggingface.co/datasets/MMMU/MMMU_Pro/tree/0d7426df4ccb3d8704a992fe6850abee7c262127/standard%20%2810%20options%29)取得 `test-00000-of-00002.parquet` 到 `_local/`。下载可以使用浏览器或 `curl -fL`；`prepare` 会按官方 LFS SHA 校验输入，冻结原始图像和官方评分代码。

```bash
python .agents/eval/mmmu-pro/skill/mmmu_pro.py prepare   --parquet _local/test-00000-of-00002.parquet --output _local/mmmu-pro-data   --revision 0d7426df4ccb3d8704a992fe6850abee7c262127   --scorer-revision 268471d0d488258990025331c7528359c324aa25   --config 'standard (10 options)' --count 32 --candidate-rows 300 --multi-image-min 4
```

先核验 QCP8 P + EP16 D 的源码、模型、量化、native制品和真实DSPARK接受率，将脱敏身份写到 `_local/runtime-identity.json`。每个正式轮前等待服务空闲、完整清P/D KV并保留JIT；相同配置依次完整跑三轮，每轮独立目录。

```bash
python .agents/eval/mmmu-pro/skill/mmmu_pro.py run   --data _local/mmmu-pro-data --output _local/mmmu-pro-w4a8-r1   --url http://SERVER_HOST:SERVER_PORT --model deepseek-v41-flash   --identity _local/runtime-identity.json   --concurrency 4 --repeats 1 --prompt-style direct --reasoning-effort none   --max-tokens 4096 --seed 968340402 --timeout 7200
python .agents/eval/mmmu-pro/skill/summarize_mmmu_pro_reruns.py   --runs _local/mmmu-pro-w4a8-r1 _local/mmmu-pro-w4a8-r2 _local/mmmu-pro-w4a8-r3   --data _local/mmmu-pro-data --output _local/mmmu-pro-w4a8-summary
```

不发送GT/explanation；不重试推理请求。固定官方 parser 的随机猜测 fallback 使用每题固定seed，另报fallback排除后的正确数；请求错误及 `length` 截断计零。这个32题集合仅为固定回归，不能当官方全量分数。记录客户端并发与服务端实际batch的区别。

## 原固定210题兼容方法

从已有制品准备原始 `mmmu-pro-210.base64.json` 和 `reference_mmmu_eval.py`；两者 SHA 必须匹配[清单](../ref/mmmu-pro-210-manifest.json)。本仓没有可验证的公开下载来源，不重造、重抽或把其他数据改名替代。原评分器作为显式输入，加载前校验SHA。

```bash
python .agents/eval/mmmu-pro/skill/mmmu_pro_210.py   --data _local/mmmu-pro-210.base64.json --reference _local/reference_mmmu_eval.py   --manifest .agents/eval/mmmu-pro/ref/mmmu-pro-210-manifest.json   --identity _local/runtime-identity.json --output _local/mmmu-210-w4a8-r1   --url http://SERVER_HOST:SERVER_PORT --model deepseek-v41-flash   --concurrency 32 --max-tokens 32768 --reasoning-effort none   --seed 968340402 --timeout 7200
python .agents/eval/mmmu-pro/skill/summarize_mmmu_pro_210_reruns.py   --runs _local/mmmu-210-w4a8-r1 _local/mmmu-210-w4a8-r2 _local/mmmu-210-w4a8-r3   --data _local/mmmu-pro-210.base64.json --reference _local/reference_mmmu_eval.py   --label w4a8-qcp8-ep16 --output _local/mmmu-210-w4a8-summary
```

三轮完整210题，原prompt、图文顺序与评分不变；temperature0/top_p1/top_k1、截断计零、零自动重试。每轮新目录，不挑最高分，不混不同版本。汇总器重建630个请求SHA并用原评分器逐条重算，报告均值、样本标准差、范围、截断和逐题波动。
