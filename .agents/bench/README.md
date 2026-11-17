# W4A8 测试

[方法与运行入口](skill/SKILL.md)维护自举命令与验收；[保留的数据](w4a8/README.md)区分历史结果和当前源码复测。这里测原生满载 step，不是 HTTP 负载或客户端 TTFT。

仅保留 DeepInfer 风格 CUSTOM 测法。依赖来自[部署环境](../deploy/README.md)，需先构建与当前源码匹配的 runtime、engine、scheduler、kernel。原始请求、日志、trace 与中间结果写 `_local/`；公开结果只放脱敏的精简数据、版本与结论。
