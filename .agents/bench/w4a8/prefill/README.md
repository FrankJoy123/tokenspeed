# Prefill 历史有效点

[JSON](results.json)和[CSV](results.csv)保留8×RTX6000D、W4A8、DP8/CP8五个有效点。DP8 4K/8K/16K为1415.93/2800.02/5583.19ms；CP8 4K/8K为310.01/486.01ms。DP8吞吐约2.89–2.93K新token/s/GPU；CP8约1.65/2.11K。

历史条件是384+32静态专家、物理槽均衡、Indexer两个原生FP4开关均1、一完整预热＋一正式轮。CP16K的Indexer配置不同，已排除；32K失败、64K未测。未完成完整曲线和平台原因取证。当前[新协议](../../skill/prefill/README.md)为R0自然路由、显式Indexer、两正式轮；历史数值只作受限参考，不证明新源码正确性或收益。
