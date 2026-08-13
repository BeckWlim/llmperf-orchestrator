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
`http://127.0.0.1:8000`，可设置 `LLMPERF_URL` 或传 `--url`。

先检查控制面：

```bash
llmperfctl health
llmperfctl scheduler status
llmperfctl planner runtime
llmperfctl provider list
llmperfctl provider models aliyun
```

`provider models ... --refresh` 会主动访问供应商目录，需要 operator 权限。目录可见
不保证推理调用成功；401 通常表示供应商 token 失效或不正确。

## 2. Runner 操作

```bash
llmperfctl runner start -f examples/test-smoke.yaml
llmperfctl runner start -f examples/test-smoke.yaml -w
llmperfctl runner status RUNNER_ID --summary
llmperfctl runner status RUNNER_ID --wait --summary
llmperfctl runner list --status failed --limit 10
llmperfctl runner logs RUNNER_ID
llmperfctl runner export RUNNER_ID -o runner.json
llmperfctl runner cancel RUNNER_ID
```

`runner start` 默认只入队并立即返回。`-w`/`--wait` 仅让 CLI 等待，不改变 Runner
持久化生命周期。CLI 中断后 Runner 继续执行，可用 ID 重连。Worker 退出码 0 不代表
基准成功；零完成请求、请求异常和首个 provider error 必须以 Runner 结果为准。

排障顺序：

1. `runner status ... --summary` 查看请求 started/completed/failed 和首错。
2. `runner logs` 查看完整 stdout/stderr、Ray Actor 异常和缺失环境变量。
3. 验证 Provider Profile 是否把 endpoint/key 注入所选 Worker。
4. 对 401 检查供应商凭据；对 404/模型错误检查精确模型 ID；对 tokenizer 错误检查
   immutable revision 解析。

## 3. Campaign 与 RunnerPlan 操作

```bash
llmperfctl campaign start -f examples/runner-plan.yaml
llmperfctl campaign start -f examples/runner-plan.yaml -w
llmperfctl campaign start -f examples/runner-plan.yaml -w \
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
llmperfctl planner preview -f examples/runner-plan.yaml
llmperfctl planner create CAMPAIGN_ID -f examples/runner-plan.yaml
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
视图；`--full` 才读取包含完整 summary 和 Worker 日志的导出文档。

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
version 3 在 `aggregate` 中使用同一口径。

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
summary、Worker 信息和捕获日志；Campaign 导出包含 aggregate、RunnerPlans 和
Runners。需要请求级记录时显式使用 `--include-requests`，避免默认传输大对象。
