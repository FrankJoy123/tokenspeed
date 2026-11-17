# Longdoc 客户端

从 [SKILL.md](SKILL.md)执行。`longdoc.py` 下载并校验冻结数据、重放请求；`longdoc_scoring.py` 保留作者确定性评分；`summarize_longdoc_reruns.py` 重建请求 SHA 并逐题重评分三轮回答。`test_longdoc*.py` 检查原图顺序、GT隔离、校验失败、评分和完整轮次统计。

Python 3.12，运行仅使用标准库；CPU测试使用 `pytest==9.1.1`。作者评分代码和许可从[ref](../ref/README.md)追溯。
