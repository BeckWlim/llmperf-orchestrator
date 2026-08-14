# LLMPerf 工程规范

## 1. 代码与架构边界

- `src/llmperf_backend/models.py`：API/YAML 严格模型与配置约束。
- `src/llmperf_backend/persistence.py`：PostgreSQL 事务、队列、状态聚合与导出。
- `src/llmperf_backend/planner.py`：驱动 RunnerPlan 编译，并从统一 Durable Dispatch 物化 Runner。
- `src/llmperf_backend/scheduler.py`：竞争 queued Runner、管理 slot 与 Worker。
- `src/llmperf_backend/worker.py`：封装 Ray task/ObjectRef、调用 benchmark 并返回结果。
- `src/llmperf_backend/safety.py`：提交前估算 Runner/请求/token/有效并发并执行硬限制。
- `src/llmperf_cli/__main__.py`：命令解析、等待轮询、日志和显示。
- `src/llmperf_cli/client.py`：HTTP 边界、认证重试和 JSON 序列化。
- `src/llmperf/cache_probe.py`、`usage.py`、`cache_analysis.py`：KV Cache P0。

保持 Planner 与 Scheduler 分工：RunnerPlan 和依赖实验先编译到统一 Durable Dispatch；
Planner 只消费到期的 `pending` Dispatch 并生产数据库中的 Runner，Scheduler 只消费
queued Runner。不要让等待时间占 Worker slot，不要从 CLI 直接计算周期或访问数据库。
跨 Runner 因果链使用 `parent_dispatch_id` 自引用 UUID 与索引寻址；Prompt Hash 只校验
实验载荷一致性，不得充当调度键。Planner 不读取父调用字段，只消费可用 Dispatch。

Worker 是一个 Runner 对应的 Scheduler 本地 Ray 执行句柄，不是 OS 子进程。Scheduler
是唯一 Ray driver，启动并守护 embedded shared runtime 或连接 external runtime。
Worker 为 Runner 提交 retry-free Ray task，并由该 task 创建至少一个 Runner-owned LLM
client actor；task 不初始化 Ray、不连接 PostgreSQL。不同 Campaign 共享 runtime 但不
共享 actor 或可变请求状态。每个 client actor 显式设置 `max_concurrency=1`，作为
Scheduler 可见的串行原子执行单元；不得改成 Threaded/Async Actor，也不得跨 Runner
复用。Actor 独立申请逻辑资源，不要求同一 Runner 的全部 Actor 同时 ready；完成的
请求槽释放 Actor 引用，让 Ray 继续调度其他 Campaign。

Campaign 并跑是核心能力。PostgreSQL claim 按各 Campaign 当前 running 数量优先选择，
并用事务级 advisory lock 防止并发 slot 在未提交快照上同时偏向同一 Campaign。每个
Worker Actor 由 Ray 细粒度排队；不得用全量 placement group 制造队首阻塞。资源争用
需要记录在结果 provenance 中，不得成为禁止 Campaign 并跑的正确性限制。

Scheduler 的动态性能守护只控制新 claim：宿主机内存达到高水位时停止领取 queued
Runner，降到较低恢复水位后再继续，形成迟滞避免抖动。守护不得把 queued 改成失败，
也不得直接杀死 running Runner；需要紧急终止时仍通过 Campaign/Runner 取消链路。
Ray Object Store 可用比例低于配置下限时也停止新 claim，恢复到较高水位后继续；Worker
完成持久化后必须丢弃 ObjectRef，避免跨 Campaign 结果长期驻留或触发磁盘 spill。
Ray actor 声明 CPU 资源且禁止基础设施自动重启/任务重试，由 Ray 集群资源调度限制
同时活跃 actor 数。Scheduler 定期读取 Ray node/cluster/available resources；健康检查
失败时停止新 claim，已有 Runner 由 Scheduler 持有的 Worker 句柄继续监管。

## 2. 数据持久化

PostgreSQL 是唯一支持的运行数据库。生产控制面、RunnerPlan 游标、Runner 队列、
请求指标和导出来源都必须可恢复、可审计。不要增加 SQLite 测试或兼容层。

当前没有生产兼容负担时可以进行直接 PostgreSQL schema 重构，但必须同步：

1. SQLAlchemy record/model；
2. `sql/postgresql/init.sql`；
3. Repository 查询与事务；
4. API/CLI/导出契约；
5. PostgreSQL 集成测试和架构文档。

避免让队列热查询随历史 Runner 无限扫描；对 queued/running 路径使用针对状态、创建
时间和 lease/claim 的索引或独立热路径，并通过查询计划验证。

## 3. 状态与错误传播

- Runner 生命周期终态是 `succeeded/failed/cancelled`。
- Campaign `status` 只表示生命周期，`outcome` 表示 Runner 聚合结果。
- Worker/Ray 异常必须传播到 Runner outcome；不得以 Ray task 正常返回覆盖请求失败。
- 零完成请求必须失败，并保存首个请求错误、HTTP code、请求计数和 Worker 日志。
- 未知 Provider cache counter 不得写成命中 0；分母未知时聚合值返回 null/不可判定。
- Runner 失败不回滚 RunnerPlan occurrence，后续轮次继续按计划物化。
- Provider 重试必须按错误类型和实验语义显式限定。Cache Prime/Warm 的模糊发送不得在
  同一 prompt family 内重试；普通负载默认不重试，以免掩盖可靠性并放大 offered load。

## 4. 测试规范

所有测试函数名最多包含三个 `_` 分隔符，例如 `test_campaign_runtime`。不要通过更长
名称编码完整场景；使用参数化 case ID 或函数内部结构表达差异。

数据库测试必须：

- 使用 `@pytest.mark.postgresql`；
- 显式请求 `postgresql_url` fixture；
- 只读取 `LLMPERF_TEST_DB`；
- URL driver 必须是 `postgresql+asyncpg`；
- 数据库名称必须包含 `test`；
- 未配置时 skip，而不是自动创建、回退 SQLite 或连接默认生产库。

运行验证：

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/pytest -q tests/test_cli.py tests/test_planner.py
.venv/bin/pytest -q
```

只有用户主动配置一次性 PostgreSQL 测试库时才运行：

```bash
export LLMPERF_TEST_DB='postgresql+asyncpg:///llmperf_test'
.venv/bin/pytest -q -m postgresql
```

保持测试去重。优先一个参数化单元覆盖同一行为矩阵；分别保留纯逻辑单元、API 合约
测试和真正 PostgreSQL 并发/事务测试，不要复制同一断言到多层。

CLI 测试必须验证统一输出边界：默认状态命令只渲染资源投影，`--json` 序列化同一投影，
`--full` 仍只渲染扩展白名单，Worker 流只由专用日志命令展示。每个命令路由必须显式
注册兼容 adapter，raw dict/list 必须被 renderer 拒绝；命令执行分支不得直接调用 JSON
renderer。

修改 CLI/API/YAML/配置/导出输入输出时必须读取并执行 `references/io.md`，将输入验证、
引用解析、投影、redaction、渲染和版本化导出作为一个跨层契约测试。

## 5. 示例与文档规范

- `examples/` 是可运行操作示例，不是测试脚本目录。
- 文件名统一采用 `example-<主要功能>.yaml`，必须以 `example` 开头，名称简洁，
  整个文件名最多包含三个 `-`。
- 示例默认配置必须在 Provider 可用时立即启动并快速得到结果。RunnerPlan、TTL 等
  周期能力应使用秒级、有界的默认值；地理时间能力的操作示例默认使用秒级相对时间表，
  把绝对地理时间表作为扩展说明。真实长周期参数只在注释或文档中说明，不得让用户
  为了验证示例等待数小时或数天。
- 当前示例基准使用 Provider `aliyun`、Model `deepseek-v4-pro`。
- 密钥只出现在环境或 `llmperf-backend config set ... --stdin`，不得提交到示例。
- 修改 YAML/API/状态语义时同步 README、`docs/ARCHITECTURE.md` 和相关中文技术报告。
- 使用当前文件名；不要在新文档中引用已经删除的 GLM 示例。

## 6. 修改与交付

1. 先用 `rg` 查找定义、消费者和测试。
2. 用 `apply_patch` 编辑，保留 dirty worktree 中无关更改。
3. 不执行 destructive git 命令，不覆盖用户已有改动。
4. 先运行聚焦测试，再运行全量测试和 `git diff --check`。
5. 未设置 `LLMPERF_TEST_DB` 时明确报告 PostgreSQL 测试已跳过。
6. 若 Backend 进程持有旧代码、Provider Profile 或环境，明确要求重启。
