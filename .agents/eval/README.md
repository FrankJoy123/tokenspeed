# 精度回归

- [Longdoc](longdoc/README.md)：MMLongBench K128 DocQA 固定50题，原始页面、4096输出预算与三轮审计。
- [MMMU-Pro](mmmu-pro/README.md)：默认使用 MMMU Tech & Engineering 固定 210 题，完整跑三轮。

通过通用 OpenAI Chat Completions HTTP 入口运行；实际 QCP8 P + EP16 D 启动从[部署入口](../deploy/README.md)进入。性能模拟接受率必须关闭。CPU 客户端与 GPU 服务依赖分离。

Longdoc 的运行身份见[共享质量身份](ref/README.md)。结果从[Longdoc W4A8](longdoc/w4a8/README.md)或[MMMU-Pro W4A8](mmmu-pro/w4a8/README.md)进入。DocQA与Agent非盲语义复核分开解释。
