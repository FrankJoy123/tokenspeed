# 图像评测客户端

从 [SKILL.md](SKILL.md)执行。

- `mmmu_pro_210.py`、`summarize_mmmu_pro_210_reruns.py`：默认固定 210 题与三轮审计。
- `mmmu_pro.py`：准备可选的官方公开 32 题子集并重放请求。
- `summarize_mmmu_pro_reruns.py`：公开子集三轮独立请求重建与重评分。
- `test_mmmu_pro.py`：题目选择与原图文顺序、GT隔离、checksum保护。

Python 3.12；公开数据准备/官方评分依赖见 `requirements.txt`，CPU测试另需 `pytest==9.1.1`。210题原评分器需要 `requests==2.34.2`（已列入清单）；原制品来源仍按SHA独立核验。
