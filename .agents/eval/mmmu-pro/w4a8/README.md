# W4A8 官方MMMU-Pro回归

QCP8/EP8 Prefill + TP1/DP16/EP16 Decode，DSPARK draft6/steps5、真实模型接受。官方MMMU-Pro standard图像固定32题/41原图，direct提示、C4、4096输出预算、seed968340402、temperature0/top_p1/top_k1、reasoning_effort=none；三完整轮、零推理重试。SMG为1.16.0.post20261115，本项成功三轮实测worker_wedge_secs=0、请求超时7200秒。

| 轮次 | 严格正确率 | 长度截断（计0） | 请求错误 |
| --- | --- | --- | --- |
| 1 | 16/32（50.0000%） | 3 | 0 |
| 2 | 17/32（53.1250%） | 1 | 0 |
| 3 | 17/32（53.1250%） | 1 | 0 |

严格准确率均值52.0833%，样本标准差1.8042个百分点。96/96请求成功，独立重建核对96份请求SHA并重评分。截断保留且严格计0，未补题或重试。每轮前后真实原生flush完成2/2 worker，17个engine load空闲，源码/native/模型身份保持一致。

本轮子集是官方固定32题。另保留的210题兼容方法未执行，不属于这轮结果，也没有并入官方32题分数。

[results.json](results.json)拥有精确三轮统计、采样、manifest/client/scorer指纹和审计计数；[共享身份](../../ref/README.md)说明实测acc2源码、与V6的字节一致性及随后V7的变更/复测边界。完整回答与运行现场不随结果发布。复现见[方法](../skill/SKILL.md)。
