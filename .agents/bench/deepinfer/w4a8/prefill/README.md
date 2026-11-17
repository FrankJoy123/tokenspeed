# W4A8 Prefill 本轮结果

本目录只保本轮当前方法产生、经两次原始记录审核的完整曲线。[测法](../../prefill/README.md)固定R0自然路由、默认算法/autotune与BF16 Indexer，采样0.3/.95/-1。

| 布局 | 新增预算/lane | encoder物理行/lane | ms/step | 新token/s/GPU | encoder行/s/GPU |
|---|---:|---:|---:|---:|---:|
|QCP8/EP8|4096|4224|358.037|1430.020|1474.709|
|QCP8/EP8|8192|8448|645.910|1585.361|1634.904|
|QCP8/EP8|16384|16896|1277.881|1602.653|1652.736|
|DP8/EP8|4096|4224|1957.707|2092.244|2157.626|
|DP8/EP8|8192|8448|3963.367|2066.929|2131.521|
|DP8/EP8|16384|16896|7979.539|2053.251|2117.415|

[QCP8 JSON](cp8.json) / [CSV](cp8.csv)和[DP8 JSON](dp8.json) / [CSV](dp8.csv)记录真实源码、依赖、模型/制品指纹、全部lane的10个间隔和分布。QCP8使用提交5fcb40a，DP8使用5e0e283；两者唯一生产差异为QCP SM120计数组初始化，核函数及算法相同。每预算独立引擎和隔离缓存、真实prime、完整warm queue和一个完整formal queue，全部输入/缓存/输出、连续满载、身份及编译增量为0均已审核。硬件相同：独占8×RTX6000D。

预算按每attention DP lane计，QCP8吞吐已除8，DP8保留全部8lane间隔。新增吞吐每请求只计4096，128行默认encoder SWA replay令encoder物理计算为4224；不能把物理行数作新增token分子。QCP8较大预算改善吞吐，DP8从4K到16K下降约1.86%，不隐藏该差异。QCP8 4K包含388.223ms慢间隔，其余点与完整分布如JSON，未裁尾或平滑。

两曲线使用TS默认portable BF16 selector，`TOKENSPEED_ENABLE_SM120_PREFILL_INDEXER`未启用，原生FP4 Indexer两个开关均关闭。质量PD部署使用显式BF16 Top512 selector，两者身份分别保存，不能将曲线当作该opt-in配置的测量。

完整日志、请求、响应、trace、绝对时钟与机器信息留在私有运行目录。诊断trace在独立代际采集，不替换正式无profiler成绩；统一性能分析见[结果入口](../README.md)。
