# 精度回归

- [Longdoc](longdoc/README.md)：MMLongBench K128 DocQA 固定50题，原始页面、4096输出预算与三轮审计。
- [MMMU-Pro](mmmu-pro/README.md)：官方公开图像子集；另保留既有210题兼容回归，其数据来源与官方 MMMU-Pro 分开说明。

通过通用 OpenAI Chat Completions HTTP 入口运行；实际 QCP8 P + EP16 D 启动从[部署入口](../deploy/README.md)进入。性能模拟接受率必须关闭。CPU 客户端与 GPU 服务依赖分离。
