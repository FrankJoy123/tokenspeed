# 共享质量身份

[w4a8-identity.json](w4a8-identity.json)记录本轮 Longdoc 和已测官方 32 题子集的共同配置、源码/模型/Tokenizer指纹、20项依赖版本、host/native SHA与原生缓存清理条件。Longdoc 的三轮成绩与采样由其 results.json 维护。固定 210 题的结果使用此前优化版 P/D 实测身份。此处不存原始题目、回答、日志或机器身份。

实测源码为`acc2c8c8e75623bbef7694eea131bee526d86cdd`，2604个生产文件与V6 `5e0e2839cf92d054adad09812fff9441b5ac947e`逐字节一致。V7 `5fcb40a13f86f6c892696bcb3f774aa0ea1e6a51`随后增加QCP所需的专用Gloo统计组启动初始化，普通非QCP DP路径保持原文；V7的全模型质量状态为未复测。本次成绩绑定实测acc2身份。

模型指纹覆盖配置、Tokenizer、encoding、索引及全部48分片的大小/头部；它不是完整tensor payload的逐字节SHA。native指纹清单以相对模块文件名排序后的紧凑UTF-8 JSON计算SHA256，无机器路径。SMG请求超时及worker_wedge_secs分别记录在数据集结果里，不能混用。

结果入口：[Longdoc](../longdoc/w4a8/README.md) / [MMMU 固定 210 题](../mmmu-pro/w4a8/README.md)。
