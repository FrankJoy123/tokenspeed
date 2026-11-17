# 精度回归

- [Longdoc](longdoc/README.md)：MMLongBench K128 DocQA 固定50题，原始页面、4096输出预算与三轮审计。
- [MMMU-Pro](mmmu-pro/README.md)：官方固定32题三轮；210兼容方法另行保留，未执行且不属本轮子集。

通过通用 OpenAI Chat Completions HTTP 入口运行；实际 QCP8 P + EP16 D 启动从[部署入口](../deploy/README.md)进入。性能模拟接受率必须关闭。CPU 客户端与 GPU 服务依赖分离。

共同配置、版本与制品指纹见[共享质量身份](ref/README.md)；当前三轮结果从[Longdoc W4A8](longdoc/w4a8/README.md)或[MMMU-Pro W4A8](mmmu-pro/w4a8/README.md)进入。DocQA与Agent非盲语义复核分开解释。
