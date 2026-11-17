# W4A8 三曲线与慢尾分析

三条正式曲线均为无profiler测量，全部慢间隔保留。原始输入、缓存、满载宽度、连续性、编译与身份已审核；Decode另由独立实现逐192个正式rank窗口、96个预热窗口复算一致。数据见[Prefill](prefill/README.md)与[Decode](decode/README.md)，诊断精确统计由[trace-summary.json](trace-summary.json)维护。

## Prefill

预算按每attention DP lane计。QCP8一个group处理该预算，DP8八个lane各处理该预算，两者全机每step新增工作量不同8倍，不能把步时比当作等总工作量加速比。每GPU吞吐已按实际工作量归一；4096新增token对应4224物理encoder行，decoder-tail计算另属模型工作，不增加新增吞吐分子。

QCP8在4K/8K/16K预算为1430.02/1585.36/1602.65新token/s/GPU；DP8为2092.24/2066.93/2053.25，平滑下降1.86%。DP8三点8lane平均步时散布仅0.00390/0.00572/0.00423ms，最大间隔/平均约1.00226/1.00178/1.00072，未见单rank漂移。没有对该轻降作硬件归因或统计等价声明。

独立QCP8-16K诊断收齐8rank CPU/GPU trace。中间6间隔共7.703253040s，各卡全stream kernel/copy/memset活动并集为98.9472%–99.2681%，最大观测空隙56.897–271.904us；这些数值不是SM occupancy或硬件饱和度。相同shape的encoder attention/index-scan在各rank的配置和耗时接近；TP0另外执行512行CED decoder-tail attention和grid(128,16,1)的分块index-scan，与[尾部选择](../../../../python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py#L923)及实际grid/calls相符。这解释rank工作构成差异，其他rank更多通信kernel驻留不能直接等同网络瓶颈。

采集8间隔均值1285.159649ms，相邻无profiler参考1276.606513ms，观测差+0.6700%；两窗覆盖不同请求，不能全归profiler。该短trace不解释4K的388.223ms慢点，也不解释DP8轻降；这些点均原样保留。

## Decode

B1/2/4/8/16/32的正式换算吞吐为151.47/245.76/394.32/579.87/646.14/705.39 token/s/GPU，未强制单调。τ=4.75来自假设接受率0.75，实际验证/保留宽度6；不能将其当真实模型接受率或质量结果。

独立诊断实际捕获180.413ms同步慢步，正常对照113.974ms。相同graph/node和launch shape的80行DeepEP dispatch驻留从1.567ms增至67.747ms；其结束、PDL copy释放及后继FC1开始均整体推迟约66.438ms，解释该窗口几乎全部增量。PDL epilogue与producer重叠，不重复相加；index-scan约18.70ms基本不变，FC1自身约48–50us。直接关键路径定位在dispatch/PDL释放，尚未区分内部initial/final barrier、GIN transfer/progress或缺失rank的GPU迟到；相同shape也不等同相同payload/自然路由负载。

全16rank原生完成间隔为180.398–180.442ms。另一次全程无CUPTI的616满载点观察在跳100点后也出现183.134ms慢尾，该尾无GC重叠；每rank仅4次gen0、最长0.077ms，gen1/gen2均0。GPU-only代际可见cgroup节流增量为0；采样线程runnable-wait远小于尾部增量。约1秒粒度的端口计数未见sampled error/retry增长，ECN/CNP并非该尾窗的突出峰值；其QP覆盖和PFC指标不完整，不能据此判定或排除拥塞。

最后一代诊断在profiler停止/导出边界发生GIN barrier timeout/CUDA719，仅15个完整GPU trace、576个共同满载点，未达到该诊断的600点要求。失败完整保留；上面的内段事件链来自失败前可核原件，不将该代当完整16rank捕获或新成绩，也不将导出失败解释为原正式无profiler慢尾原因。原正式各慢峰没有逐一录到，内部等待源仍未定，不能声称所有底层原因已解决。

## 验证与适用范围

[现有Elastic数值用例](elastic-validation.json)在实际两节点16rank完成原样9case、144次rank-case检查，严格0容差、零失败/错误/跳过，覆盖改变输入和路由的CUDA graph replay及target/draft/target复用。该测试观测到GIN_IB_GDAKI；它只属于本次算子验证，不倒推旧bench的provider/HCA，也不替代模型质量回归。

原始trace和官方Perfetto v59.1导入均保留。CPU complete-event层级重叠告警未隐藏，GPU类别计数与区间并集另行核验；不依据模糊CPU父子层级作因果推断，没有UI或硬件计数器占用率验证。请求的NIC环境映射、NCCL允许列表与实际绑定分开记录，初始化日志未记录的选择不补写。所有诊断均不替换正式分数，生产算法与通信设置保持固定。
