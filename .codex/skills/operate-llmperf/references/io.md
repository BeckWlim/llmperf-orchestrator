# LLMPerf 输入输出管理契约

## 1. 输入管线

所有外部输入统一经过以下阶段，不得在命令或 endpoint 中旁路：

```text
CLI/YAML/JSON/environment
  -> decode
  -> strict validate
  -> resolve Backend-owned references
  -> safety assess
  -> atomic persist
```

- YAML 只用安全解析器读取，并要求顶层 mapping；未知字段由严格 Pydantic 模型拒绝。
- CLI 只负责文件、参数和 HTTP 边界，不直接访问 PostgreSQL，也不自行展开调度语义。
- Provider URL、API key、数据库凭据和私钥只进入 Backend/CLI 配置；workload 只携带稳定 ID。
- Secret 值只通过 `--stdin`、权限受控配置文件或进程环境输入，不进入参数、日志、YAML、
  metadata、summary 或导出。
- Tokenizer/Dataset 等远端引用在入队前解析为 Backend-owned 本地 artifact 和不可变版本；
  Worker 只接收已选择 Provider 的最小环境及本地路径。
- 在一个数据库事务中创建完整 Campaign/RunnerPlan/Dispatch 图；任一解析或安全检查失败
  时不保留部分工作负载。
- 为文件大小、请求数、token、并发、超时、计划 occurrence 和 artifact 解析设置显式边界。

## 2. 输出管线

所有查询输出统一经过：

```text
PostgreSQL/API authoritative record
  -> command compatibility adapter
  -> resource projector
  -> CLIProjection
  -> centralized renderer or versioned export
```

- `execute`/HTTP client 只返回结构化数据，不直接打印。
- CLI 的每个 `command.subcommand` 都必须在兼容层显式注册 adapter；adapter 负责旧/新字段
  别名、缺省值和报文形状检查，再调用资源 projector。禁止 identity adapter、通用 raw
  fallback 或未注册路由继续执行。
- 每种资源维护一个 projector，白名单选择稳定、可操作字段；不得先复制完整记录再删除
  少数字段。renderer 只接受 `CLIProjection`，传入 `dict`/`list` 原始响应必须失败。
- 默认 `status/list/health` 输出人类可读的轻量投影，不输出原始 JSON。
- 显式 `--json` 序列化同一轻量投影，字段语义与默认文本一致。
- `--full` 只扩大到显式登记的详细兼容投影，仍禁止输出完整 API 记录；完整且大型结果只通过
  带版本的 export 文件取得。
- Worker stdout/stderr 只由专用 `logs` 命令或显式完整导出显示。
- `start/cancel/export` 默认不向 stdout 倾倒响应；进度、durable ID 和操作信息写入 stderr。
- `render_result` 是 CLI 唯一展示策略入口；禁止命令分支调用
  `print_json(raw_response)` 或自行拼接另一套投影。

## 3. 数据最小化与安全

Projector 默认过滤：凭据、Authorization、私钥路径、数据库 URL、内部配置绝对路径、
密钥 ID/轮换细节、完整 Worker 流、原始请求正文、Prompt 文本、大型嵌套 summary，以及
不稳定的内部实现字段。需要诊断时通过授权 endpoint、扩展白名单 `--full`、专用 logs 或版本化导出
显式取得；CLI 投影不是 Backend 授权和 API redaction 的替代品。

错误默认输出简洁分类、HTTP/Provider code、首个可操作消息和 durable ID。完整 traceback
保留在服务 journal 或 Worker logs，不在普通状态命令中倾倒。所有日志和导出继续执行
secret redaction 与大小上限。

## 4. 命令模式矩阵

| 模式 | stdout | stderr | 数据范围 |
|---|---|---|---|
| 默认查询 | 稳定文本投影 | 操作日志 | 白名单字段 |
| `--json` | 同一投影 JSON | 操作日志 | 白名单字段 |
| `--full` | 详细兼容投影 | 操作日志 | 扩展白名单诊断字段 |
| `logs` | 明确分隔的 Worker 流 | 操作日志 | 有界 stdout/stderr |
| `export -o` | 默认静默 | 文件位置/结果 | 版本化导出文件 |
| start/cancel | 默认静默 | ID、状态变化 | 操作摘要 |

`health` 默认只投影 Backend、Database、Planner、Provider 数量和 Auth 健康状态；过滤
`config_source`、配置代次、active key ID 和轮换内部状态。`health --json` 输出同一投影，
`health --full` 仍输出兼容投影，不得包含完整健康响应或内部配置字段。

## 5. 变更检查清单

修改输入或输出时同时完成：

1. 更新严格模型或输入 decoder，并增加无效、未知、越界输入测试；
2. 更新资源 projector，确认默认结果不包含 raw record、secret、日志或大型嵌套字段；
3. 为每个新增命令注册 adapter；只在集中 renderer 接入默认、`--json`、`--full` 和 action
   策略，且没有通用回退；
4. 更新 CLI help、Skill 参考和必要的 API/export 版本说明；
5. 测试文本投影与 JSON 投影语义一致，`--full` 仍经白名单，raw payload 被 renderer 拒绝，
   stderr/stdout 不串流；
6. 对真实响应样本执行敏感字段和体积审计，再运行聚焦与全量测试。
