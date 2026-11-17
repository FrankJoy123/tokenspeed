# 冻结依据

- `longdoc-k128-regression50.json`：固定50题，按 `SHA256("dsv41-k128-regression50-v1:" + id)` 从冻结200题选取，按ID排列，不读取模型分数。
- `longdoc-k128-subset200.json`：父题集，完整字节 SHA 为 `83c38b6f67fdc25d9a09d05047f5f6b936d2c2713734fd1fd457d3a600987b17`。
- `sources.json`：作者评分代码精确提交、URL和源文件 SHA；`MMLONGBENCH_LICENSE.txt`：MIT许可。

数据：[ZhaoweiWang/MMLongBench](https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/tree/746b5c42518390e5c3b71d168d0dc49c78824142)，原始页面来源及revision由两份清单的 `images` 固定。`prepare` 校验JSONL原件、所选每行及下载后原图库存SHA。下载数据不随Git分发，按源数据许可使用。

评分源码：[EdinburghNLP/MMLongBench utils.py](https://github.com/EdinburghNLP/MMLongBench/blob/a857dffa00de3d78035ac6fd454738254f4e1620/utils.py)。本目录保存来源，实际裁剪的纯确定性评分实现位于 `../skill/longdoc_scoring.py`。
