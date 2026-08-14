# llmperfctl 操作规范

## 1. 启动前检查

Backend 使用 PostgreSQL，不支持 SQLite：

```bash
llmperf-backend config set DATABASE_URL postgresql+asyncpg:///llmperf
llmperf-backend config set LLMPERF_PROVIDER_ALIYUN_URL \
  https://dashscope.aliyuncs.com/compatible-mode/v1
printf '%s' "$ALIYUN_API_KEY" | \
  llmperf-backend config set LLMPERF_PROVIDER_ALIYUN_KEY --stdin
llmperf-backend config set LLMPERF_DEFAULT_PROVIDER aliyun
llmperf-backend config list
llmperf-backend
```

配置加载优先级：进程环境变量、`LLMPERF_ENV_FILE`、用户持久化配置、当前目录
`.env`。Provider Profile、密钥或数据库配置改变后重启 Backend。CLI 默认连接
`http://127.0.0.1:8000`，并独立读取 `~/.config/llmperf/cli.env`。使用
`llmperfctl config set/get/unset/list/path` 管理该文件；`LLMPERF_CLI_ENV_FILE` 可指定
其他路径。CLI 显式参数优先于进程环境，进程环境优先于 `cli.env`。

先检查控制面：

```bash
llmperfctl health
llmperfctl health --json
llmperfctl scheduler status
llmperfctl planner runtime
llmperfctl provider list
llmperfctl provider models aliyun
```

`scheduler status` 报告 `ray_mode=embedded|external` 和 `ray_runtime` 健康/资源快照。
每个 Runner 必须通过 Ray actor 执行。`scheduler.ray_address` 为空时 Scheduler 只启动
一套 embedded shared runtime；配置地址时连接 external runtime。Scheduler 是唯一 Ray
driver；Worker 是 Ray task + ObjectRef 的执行句柄，不是子进程。调大 slot 前同时
核对 `ray_num_cpus / ray_actor_num_cpus` 的 actor 容量。

`provider models ... --refresh` 会主动访问供应商目录，需要 operator 权限。目录可见
不保证推理调用成功；401 通常表示供应商 token 失效或不正确。

Ubuntu 部署模板和指引统一放在仓库 `deploy/systemd/`，该目录不是 systemd 的运行加载
目录。准备好 `.venv`、PostgreSQL schema 和运行用户的 Backend 配置后，把
`llmperf-backend.service.template` 中的项目路径、用户和组占位符渲染到
`/etc/systemd/system/llmperf-backend.service`，再由系统级 `sudo systemctl` 启动。
unit 属于系统级服务，但 Backend 进程必须以模板指定的普通用户运行。服务直接执行
现有 `.venv/bin/llmperf-backend`，不重复声明 Backend 已能自行解析的环境变量，并通过
systemd cgroup 管理 Backend 与 embedded Ray 子进程。Provider 密钥仍只保存在该用户
自有的 `0600` 配置中。

## 2. Runner 操作

```bash
llmperfctl runner start -f examples/example-smoke.yaml
llmperfctl runner start -f examples/example-smoke.yaml -w
llmperfctl runner status RUNNER_ID --summary
llmperfctl runner status RUNNER_ID --wait --summary
llmperfctl runner list --status failed --limit 10
llmperfctl runner logs RUNNER_ID
llmperfctl runner export RUNNER_ID -o runner.json
llmperfctl runner cancel RUNNER_ID
```

`runner start` 默认只入队并立即返回。`-w`/`--wait` 仅让 CLI 等待，不改变 Runner
持久化生命周期。启动、等待、取消和导出命令默认不向 stdout 输出响应 JSON，只把状态
变化写入 stderr；需要结果时使用 `status/list/logs` 或显式导出。CLI 中断后 Runner
继续执行，可用 ID 重连。Ray Worker task 正常返回不代表基准成功；零完成请求、请求
异常和首个 provider error 必须以 Runner 结果为准。

排障顺序：

1. `runner status ... --summary` 查看请求 started/completed/failed 和首错。
2. `runner logs` 通过专用日志接口查看完整 stdout/stderr、Ray Actor 异常和缺失环境变量。
3. 验证 Provider Profile 是否把 endpoint/key 注入所选 Worker。
4. 对 401 检查供应商凭据；对 404/模型错误检查精确模型 ID；对 tokenizer 错误检查
   immutable revision 解析。

## 3. Campaign 与 RunnerPlan 操作

```bash
llmperfctl campaign start -f examples/example-runner-plan.yaml
llmperfctl campaign start -f examples/example-runner-plan.yaml -w
llmperfctl campaign start -f examples/example-runner-plan.yaml -w \
  -o campaign-report.json
llmperfctl campaign status CAMPAIGN_ID
llmperfctl campaign status CAMPAIGN_ID --json
llmperfctl campaign status CAMPAIGN_ID --full --include-requests
llmperfctl campaign list
llmperfctl campaign export CAMPAIGN_ID -o campaign-report.json
llmperfctl campaign cancel CAMPAIGN_ID
```

Planner 控制命令：

```bash
llmperfctl planner preview -f examples/example-runner-plan.yaml
llmperfctl planner create CAMPAIGN_ID -f examples/example-runner-plan.yaml
llmperfctl planner list --status active
llmperfctl planner status RUNNER_PLAN_ID
llmperfctl planner events RUNNER_PLAN_ID
llmperfctl planner pause RUNNER_PLAN_ID
llmperfctl planner resume RUNNER_PLAN_ID
llmperfctl planner cancel RUNNER_PLAN_ID
```

Campaign 创建会先解析全部 Provider、Tokenizer 和 Dataset 依赖，再在一个事务中创建
Campaign、入队即时 Runner、注册 RunnerPlan；任一校验失败不应留下部分 Campaign。
`campaign status` 默认输出 Campaign 生命周期/结果总览，以及按时间排序的每个 Runner
ID、状态、Provider/Model 和 started/completed/failed 请求摘要。`--json` 输出同一份轻量
视图；`--full` 输出扩展白名单投影。完整 summary、请求记录和 Worker 日志应通过版本化
export 或专用 `runner logs` 取得，不得绕过兼容适配器直接打印。

`campaign cancel` 在一个 PostgreSQL 事务中取消 active/paused RunnerPlan、blocked/
pending Dispatch、未终态协议实例和 queued Runner；running Runner 设置取消请求，
Scheduler 在下一次轮询发现后先 `ray.cancel`，并在 `cancel_grace_seconds` 后强制取消；
Worker task 结束后派生 Actor 引用随执行槽释放。
已经终态的 Runner 和结果不回滚。若 Backend 已停止，应在恢复 Scheduler 前先确保
Campaign 已取消，否则 durable queued/pending 工作会在服务恢复后继续派发。

## 4. 状态解释

Campaign 有两个正交聚合维度：

- `status`：生命周期，取 `planned/queued/running/paused/completed/cancelled/empty`。
- `outcome`：执行结果，取
  `pending/succeeded/partial_failed/failed/cancelled/no_runs`。
- `has_failures`：是否存在失败 Runner。

例如 8 个 Runner 中 7 成功、1 失败：

```json
{
  "status": "completed",
  "outcome": "partial_failed",
  "has_failures": true
}
```

不要把它标成生命周期 `failed`。`-w` 在 Campaign 生命周期终态结束；
`partial_failed`、`failed` 或 `cancelled` 仍令 CLI 返回退出码 2。Campaign export
version 5 在 `aggregate` 中使用同一口径，并附带 `protocol_definitions`、
`protocol_instances`、`dispatches` 与 `protocol_analyses`。协议实例处于父调用完成、
子调用尚未到期的等待期时，Campaign 状态保持 `planned`。
`cache-residency/v1` 的分析会同时保留地理时间表、计划 offset、实际 Prime-to-Warm
delay，并标记为 `access_conditioned_residency`；不要把它解释为被动 TTL。

## 5. 等待与日志

`campaign start -w` 默认每 2 秒执行一轮 HTTP 轮询，每轮读取 Campaign 聚合、相关
RunnerPlans 和相关 Runners。只在快照变化时打印 INFO 日志。可用：

```bash
llmperfctl --log-level debug campaign start -f FILE -w \
  --poll-interval 2 --timeout 3600
```

Backend 内 Planner 和 Scheduler 默认各自每 1 秒轮询 PostgreSQL，与 CLI 观察轮询
独立。等待计划不占 Scheduler slot。退出等待 CLI 不会取消 Campaign。

日志保持以下边界：

- stderr：进程信息、HTTP 调试元数据、状态变化、Worker 进度。
- stdout：JSON 或紧凑表格，便于重定向和脚本处理。
- `runner logs`：Backend 持久化的 Worker stdout/stderr。
- 任何日志均不得包含 Provider key、Bearer token 或私钥。

## 6. 导出与结果定位

核心结果首先写入 PostgreSQL；JSON 是导出视图，不是唯一存储。Runner 导出包含
summary、Worker 信息和捕获日志；Campaign 导出包含 aggregate、RunnerPlans、
Protocol Definitions、Protocol Instances、Dispatches、Protocol Analyses 和 Runners。需要请求级记录时显式使用
`--include-requests`，避免默认传输大对象。

## 7. 统一输出处理策略

CLI 严格保持四层边界：`execute` 只返回结构化 API 数据，按 `command.subcommand` 注册的
兼容 adapter 检查报文形状并处理版本差异，资源 projector 生成稳定白名单，
`render_result` 是唯一展示策略入口。命令分支不得直接打印 API 原始响应。

- `status/list` 默认输出人类可读的稳定投影，不输出 JSON。
- 显式 `--json` 输出同一轻量投影，字段语义必须与默认文本一致。
- 显式 `--full` 也只能输出扩展白名单，不是 raw response 逃生口。
- Worker stdout/stderr 只由 `runner logs` 或完整导出呈现。
- `start/cancel/export` 默认不向 stdout 输出响应体；进度、ID 和结果位置写入 stderr 日志。
- 新增资源命令时先注册 adapter 和 projector，再接入集中 renderer，并同时测试默认、
  JSON 和 full 边界；禁止 identity adapter、未注册路由和 `print_json(raw_response)` 旁路。

Runner 的统一投影入口由兼容层调用 `project_runner`（`summarize_runner` 保留为兼容别名）。
因此 `runner status ID` 与旧的
`runner status ID --summary` 等价；`runner status ID --json` 输出该投影的 JSON；只有
`runner status ID --full` 输出扩展白名单。Campaign 和集合命令遵循相同规则。

`health` 同样必须经过统一投影：默认文本和 `--json` 只包含 Backend/Database/Planner、
Provider 数量及 Auth 健康状态；内部配置路径、配置代次和密钥轮换细节只允许通过显式
的授权诊断 API 查看，`--full` 也不得直出。项目级输入输出契约见 [io.md](io.md)。
