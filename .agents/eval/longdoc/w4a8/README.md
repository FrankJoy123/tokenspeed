# W4A8 Longdoc回归

QCP8/EP8 Prefill + TP1/DP16/EP16 Decode，DSPARK draft6/steps5、真实模型接受。固定50题、C4、4096输出预算、seed968340402、temperature0/top_p1、reasoning_effort=none，三完整轮、零推理重试。SMG为1.16.0.post20261115，本项实测worker_wedge_secs=3、请求超时7200秒。

| 评分 | 三轮（%） | 均值 ± 样本标准差（百分点） |
| --- | --- | --- |
| 确定性DocQA，截断严格计0 | 60.1252 / 61.7360 / 60.1968 | 60.6860 ± 0.9100 |
| Agent语义复核，非盲评 | 67.7599 / 67.2599 / 64.1299 | 66.3832 ± 1.9674 |

150/150请求成功、0错误、0截断；独立重建核对150份请求SHA并重评分。实际prompt长度47872–63342 tokens，原图、页面顺序与固定清单保持一致。每轮前后在全部17个engine load空闲后执行真实原生flush，2/2 worker完成，保留JIT；源码/native/模型身份保持一致。

语义复核使用同150份响应，去重66候选；44个仅在题目、标准答案和输出原字节相同的条件下复用判断，22个由Agent直接逐项判分，无新增推理请求。复核Agent已见运行身份和DocQA总分，属于非盲复核；两个指标分别报告，不合并成一个分数。

[results.json](results.json)拥有精确三轮统计、采样、manifest/client/scorer指纹和审计计数；[共享身份](../../ref/README.md)说明实测acc2源码、与V6的字节一致性及随后V7的变更/复测边界。完整回答与运行现场不随结果发布。复现见[方法](../skill/SKILL.md)。
