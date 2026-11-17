# W4A8 EP16 Decode

[ep16.csv](ep16.csv)提供六档曲线及双轮rank0步时；[ep16.json](ep16.json)保留每轮全部16rank的整数纳秒间隔、分布、全局共同端点和脱敏身份。仅含当前协议的新测量。

每请求131072token=124416实际缓存+6656唯一后缀；采样0.3/.95/-1、R0自然路由、BF16 Indexer。单引擎/公共前缀跨B复用，每B先200个真实全16rank满载步骤。两轮分别从同一global dispatch step上的全16实际width6交集首个连续序列，固定跳100点再取51共同端点，形成各50间隔。所有慢点保留，无按时长选窗或单调处理。

网络身份只记录本次requested NIC map、NCCL HCA allowlist、GID及native依赖；未采INIT/NET provider或实际逐rank HCA选择，不能倒推ElasticBuffer逐rank绑定，后续qualification日志也不回填旧测量。边界见JSON network_identity_scope。

实测源码为JSON measured_source中的V6；最终跨rank边界审计采用后续修正的audit.py/analysis.py，文件SHA在reanalysis中单独记录，不能把它归为V6原本的本地preview行为。修正只重算不可变原件，不重跑模型、不覆盖原始日志。已与独立实现逐一核对192个formal窗口和96个warm窗口；只覆盖单节点的preview始终不是最终验收。

主曲线使用rank0两轮共100间隔均值；tokens_s_gpu按DeepInfer假设τ=1+5×0.75=4.75换算，0.75不是实测接受率。全部rank分布用于诊断，不能把相关的16rank间隔当作独立统计样本。同步慢尾仍保留，其原因须由单独diagnostic trace区分；无profiler正式窗口未被profile污染。方法见[Decode](../../decode/README.md)。
