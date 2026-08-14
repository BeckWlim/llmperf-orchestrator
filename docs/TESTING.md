# 测试边界

测试按责任边界组织，避免同一行为在多个模块重复断言：

| 模块 | 负责验证 | 不负责验证 |
|---|---|---|
| `test_backend_worker.py` | Worker 的 Ray task/ObjectRef 封装、Actor 资源、环境隔离、结果 outcome | PostgreSQL 事务、Scheduler 队列轮询 |
| `test_backend_scheduler.py` | 单一 Ray runtime 所有权、Scheduler→Runner→Worker 装配、心跳/取消交接 | benchmark 指标算法、真实数据库并发 |
| `test_backend_safety.py` | workload 容量估算、Actor 预算、宿主机内存与 Object Store 水位约束 | Ray 实际调度、Provider 行为 |
| `test_benchmark.py` | 请求执行算法、指标归一化、cache probe 依赖与统计 | Backend 生命周期和持久化 |
| `test_sql.py` | PostgreSQL 状态机、事务并发、跨 Campaign 公平 claim | 无数据库 fallback、Provider 网络请求 |
| `test_cli.py` | HTTP 命令契约及 `Worker` 名称、可空 PID、日志显示兼容性 | Scheduler/Ray 内部实现 |

`Worker` 是稳定的领域名称，不再表示 OS 子进程。旧配置中的 `worker_module`、
`working_directory`、`cancel_grace_seconds` 和 `log_bytes_limit` 继续被接受；其中
`cancel_grace_seconds` 和 `log_bytes_limit` 仍用于 Ray Worker 取消与日志控制，前两项仅作
配置/API 兼容。Runner 的 `worker.process_id` 保持可空，新的 Ray task 标识记录在完整
summary 的 `execution_runtime.worker_id` 中。

默认测试不得连接数据库或 Provider：

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/pytest -q
```

PostgreSQL 语义只在用户显式提供一次性测试库时运行：

```bash
export LLMPERF_TEST_DB='postgresql+asyncpg:///llmperf_test'
.venv/bin/pytest -q -m postgresql
```
