---
name: longdoc-regression
description: 准备并执行 MMLongBench K128 DocQA 固定50题三轮回归，审计请求与确定性评分。
---

# Longdoc 固定50题

在仓库根目录执行，Python 3.12，无额外运行依赖。数据放 `_local/longdoc-data`；下载 URL、revision、逐行 SHA 和原图库存 SHA 由[冻结清单](../ref/longdoc-k128-regression50.json)固定。

```bash
python .agents/eval/longdoc/skill/longdoc.py prepare --data _local/longdoc-data
```

确认 QCP8 P + EP16 D 使用本次源码及匹配native制品、真实DSPARK接受率；等待空闲后清 P/D KV，保留 JIT。记录源码提交、模型配置/量化清单 SHA、加载模块与native文件SHA到 `_local/runtime-identity.json`。每轮前重新核验身份和缓存状态。

```bash
python .agents/eval/longdoc/skill/longdoc.py run   --data _local/longdoc-data --output _local/longdoc-w4a8-r1   --url http://SERVER_HOST:SERVER_PORT --identity _local/runtime-identity.json   --model deepseek-v41-flash --concurrency 4 --max-tokens 4096 --timeout 7200
```

相同服务配置完整执行 r1、r2、r3，每轮新输出目录；固定 temperature0、top_p1、seed968340402、reasoning_effort=none，允许提前EOS。无推理请求重试；累计3条错误后停止追加，不完整轮不能当正式结果。数据下载的有界重试不改变推理零重试协议。不要用 `--limit` 的 smoke 代替正式50题。

```bash
python .agents/eval/longdoc/skill/summarize_longdoc_reruns.py   --runs _local/longdoc-w4a8-r1 _local/longdoc-w4a8-r2 _local/longdoc-w4a8-r3   --data _local/longdoc-data   --manifest .agents/eval/longdoc/ref/longdoc-k128-regression50.json   --output _local/longdoc-w4a8-summary
```

汇总须核对150条请求与原图哈希、原评分重算、均值/样本标准差/范围、逐题波动、错误、截断和真实token usage。4096输出预算与作者128预算不同，不混作同协议比较。扩展200题时显式选择 `../ref/longdoc-k128-subset200.json`，独立准备和执行，不静默换题。

```bash
python -m pytest -q .agents/eval/longdoc/skill
```
