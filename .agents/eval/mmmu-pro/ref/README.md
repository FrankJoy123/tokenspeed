# 来源与协议身份

`MMMU/MMMU_Pro` 数据 revision：`0d7426df4ccb3d8704a992fe6850abee7c262127`；官方评分代码 revision：`268471d0d488258990025331c7528359c324aa25`。精确下载链接和 SHA 在 `sources.json`；`official-evaluate.py.gz`、`official-prompts.yaml.gz` 是 gzip 封装的固定官方原字节，解压后 SHA 对照 `sources.json`；`official-LICENSE` 是同提交 Apache2.0 许可原件，适用于这两份引用源码。gzip封装保护原件不被格式化器改写。准备脚本从同一公开revision下载评分原件并记录SHA，不使用浮动main。

`mmmu-pro-210-manifest.json` 只固定历史兼容210题数据/评分器 SHA 和题号，不含原图或回答。该集合是 MMMU Tech & Engineering 子集，不是官方 MMMU-Pro 全集；无法从上述公开parquet直接恢复。没有公开来源前，原制品是必需输入，不能声称 `git clone` 即具备210题数据。

数据与图像按原数据集许可取得；本仓只保留必要方法与许可，不分发原始评测数据。
