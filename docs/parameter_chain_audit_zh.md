# 参数链路审计

本文档是公开版参数链路摘要，不包含历史私有对照材料。

## 当前对称化字段

- `phono3py_symmetrize_fc2`：控制 finite-displacement 热输运路径中是否调用
  Phono3py 官方 FC2 力常数对称化。
- `phono3py_symmetrize_fc3`：控制 finite-displacement 热输运路径中是否调用
  Phono3py 官方 FC3 力常数对称化。
- HiPhive FC3 拟合路径会记录用户是否请求 Phono3py FC2/FC3 对称化，但该路径不
  调用这些 Phono3py 钩子。

## 已废弃兼容项

旧输入别名 `phono3py_fc2_asr` 已废弃，仅用于兼容旧配置读取，并会映射到
`phono3py_symmetrize_fc2`。新的 CLI 命令和文档应使用
`phono3py_symmetrize_fc2`。
