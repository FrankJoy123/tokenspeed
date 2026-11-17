# Longdoc K128 DocQA

固定50题回归来自 MMLongBench 的 K128 DocQA 构造档位：39份文档、每题48–102张原始页面、1952张不同图。K128 是作者构造档位，实际输入长度以服务 usage 为准。客户端保留原图和顺序，不缩图、不删页、不发送答案或证据页。

- [运行方法](skill/SKILL.md)与[脚本说明](skill/README.md)。
- [冻结清单、来源与许可](ref/README.md)。
- [W4A8 结果边界](w4a8/README.md)。

主分数沿用冻结的确定性 `doc_qa`：数字容差、列表部分分、字符串 F1；另报截断计零的 strict 分数。此方法不是原始 MMLongBench-Doc 的三阶段 LLM judge，也不能与该完整基准成绩混用。
