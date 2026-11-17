# MMMU-Pro 与兼容回归

- [运行方法](skill/SKILL.md)：公开官方 MMMU-Pro 图像子集可自举；另有原固定210题兼容入口。
- [脚本](skill/README.md)、[来源与冻结清单](ref/README.md)、[W4A8 结果](w4a8/README.md)。

**两种协议分开解释。** 官方公开方案来自 `MMMU/MMMU_Pro` 的 `standard (10 options)`，固定首300候选、32题及至少4题多图，使用固定官方 parser。既有文件名 `mmmu_pro_210.py` 对应 MMMU Tech & Engineering 固定子集（191选择、19开放、6题多图），并非官方 MMMU-Pro 全集；它要求原始数据和原评分器制品，不能拿公开32题替代历史210题回归。
