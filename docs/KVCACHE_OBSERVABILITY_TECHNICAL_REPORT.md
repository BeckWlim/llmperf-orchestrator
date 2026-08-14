# LLMPerf 外部 KV Cache 可观测性技术报告

> 状态：P0 实施版  
> 日期：2026-08-13  
> 适用仓库：`llmperf-orchestrator`  
> 来源：由 `~/work/llmperf-kvcache-observability-tutorial.md` 重构为面向设计、实施与验收的中文技术报告。

## 1. 摘要

本报告讨论如何仅通过生产大模型服务的外部请求，识别跨请求前缀 KV Cache 是否存在、命中范围有多大，以及缓存对首 Token 延迟（TTFT）的影响。结论如下：

1. 外部黑盒测试可以形成强证据，但不能证明服务端内部缓存实现细节。最可靠的证据来自供应商命中计数、严格配对的 TTFT 变化，以及前缀突变实验的一致性。
2. 单纯重复 Prompt 或比较全局平均值不足以得出缓存结论。实验必须预先生成确定性请求计划，并保证每个 family 的 prime 请求成功后才发送 warm 请求。
3. `usage`、流式时间点、请求顺序、Tokenizer 来源和 Provider 请求标识必须作为可审计原始数据保存；汇总值应从这些数据派生，而不是替代它们。
4. 当前 P0 不需要立即修改数据库表结构。现有 `benchmark_request_results.metrics` 和 `benchmark_runners.summary` JSONB 足以容纳仍在演进的指标协议。对于高频检索、排序和跨运行关联的字段，建议在 P1 稳定后列化并建立索引。
5. P0 已在当前框架内形成完整落地路径：请求计划、依赖调度、Usage 归一化、缺失值安全聚合、配对统计、时序与来源记录、Tokenizer 防护，以及正确命名的流式指标。

## 2. 范围与非目标

本报告关注两类问题：

- 外部 KV Cache 的存在性、命中计量和性能收益验证；
- LLMPerf 为产生可信结论所需的数据、调度、统计和持久化能力。

以下内容不属于 P0：

- 通过外部请求恢复服务端 KV Block、Page Table 或显存布局；
- 精确区分所有内部排队、路由、批处理和调度策略；
- 跨 Worker、跨进程的分钟级或小时级 TTL 持久化调度；
- 用 SQLite 验证数据库行为；生产持久化仍以 PostgreSQL/JSONB 设计为准；
- 将统计相关性表述为服务端内部机制的确定性证明。

## 3. 观测对象与证据边界

### 3.1 两类 KV Cache

一次请求内部的自回归解码缓存用于避免每生成一个 Token 都重新计算既有上下文。这是推理实现的基本机制，但它不等同于跨请求 Prompt Cache。

本报告观测的是跨请求前缀缓存：如果新请求与历史请求具有可复用的 Token 前缀，服务端可能跳过部分 Prefill 计算。其收益通常体现为：

- warm 请求的 TTFT 下降；
- Provider Usage 中出现 cached/hit tokens；
- Prompt 早期突变破坏命中，晚期突变保留部分命中；
- 随着间隔或缓存压力增加，收益逐步衰减。

### 3.2 TTFT 的近似分解

外部看到的 TTFT 可写为：

\[
TTFT = T_{network} + T_{gateway} + T_{queue} + T_{prefill} + T_{schedule} + T_{first\ decode}
\]

前缀缓存主要减少 `T_prefill`，但路由、排队、批处理和网络波动都可能掩盖它。因此，仅观察一次 TTFT 下降不能证明缓存命中。

### 3.3 黑盒可推断内容

在控制变量充分时，外部实验能够判断：

- 是否观察到跨请求复用证据；
- 可复用前缀长度与 TTFT 收益是否相关；
- 缓存收益是否受到并发、路由、TTL 或容量压力影响；
- Provider 的缓存计数是否与延迟变化一致。

外部实验通常不能唯一确定：

- 缓存是实例级、Pod 级、节点级还是全局共享；
- 是精确 Token 前缀、Block Hash、Radix Tree 还是其他实现；
- 缓存未命中究竟源于过期、淘汰、路由变化还是容量不足；
- 服务端实际调度队列、Batch 构成和 KV 占用量。

因此结果必须使用“已确认外部证据”“计量确认”“延迟推断”“未观察到”和“证据不足”等分级表述。

## 4. 证据等级

从强到弱，推荐使用以下证据层级：

| 等级 | 证据 | 解释 |
|---|---|---|
| A | Provider 返回合法且覆盖率足够的 cache hit/miss 计数，并与配对 TTFT 改善一致 | 最强外部证据 |
| B | Provider 返回 cache hit 计数，但延迟置信区间不显著 | 计量确认，性能收益未确认 |
| C | 无计数，但 prime/warm 配对 TTFT 差异的 Bootstrap 区间稳定为正 | 延迟推断 |
| D | 仅有重复 Prompt 的平均 TTFT 更低 | 易受路由、负载和顺序偏差影响 |
| E | 单次请求或非配对全局统计 | 不足以支持结论 |

P0 对应的机器判定为：

- `confirmed_external`：命中计数与配对延迟均支持缓存存在；
- `accounting_confirmed`：命中计数支持，但延迟证据不足；
- `latency_inferred`：无命中计数，配对延迟支持；
- `not_observed`：计数覆盖充分但没有观察到命中；
- `inconclusive`：数据质量、覆盖率或样本量不足。

这些 Verdict 是证据强度摘要，不是服务端内部实现声明。

## 5. 实验设计

### 5.1 冻结环境

每个可比较 Runner 至少要固定并持久化：

- Provider、模型标识与 Endpoint；
- Tokenizer ID、请求 Revision、解析后的不可变 Revision、选择来源和准确度；
- Sampling 参数，包括温度、最大输出 Token 和流式 Usage 开关；
- 输入 Token 分布、输出 Token 分布、并发度和 Dataset Seed；
- Cache Probe 模式、family 数、warm 重复次数和计划种子；
- 客户端版本、运行时间、Runner ID 与安全的 Provider Request ID。

禁止把 API Key、Authorization Header 或完整敏感响应头写入结果。

### 5.2 配对 Prompt Family

一个 family 至少包含：

- `prime`：用于建立候选缓存状态；
- `warm`：在 prime 成功后发送，用于观测复用。

P0 支持四种模式：

| 模式 | prime/warm 关系 | 用途 |
|---|---|---|
| `exact_repeat` | 完全相同 | 缓存存在性测试 |
| `shared_prefix` | 指定 Token 边界前相同，边界处突变 | 验证共享前缀长度 |
| `early_mutation` | Prompt 前部 Token 突变 | 验证早期前缀破坏 |
| `late_mutation` | Prompt 后部 Token 突变 | 验证部分前缀复用 |

突变在 Token ID 层完成，并要求 decode/encode 往返后 Token 序列保持精确。请求计划在发流量前一次生成，使用固定 Seed，并为每个请求记录：

- `request_id`、`family_id`、`role`、`occurrence`；
- `plan_index`、实际 `dispatch_index`、`completion_index`；
- Prompt Hash、本地 Token 数、预期共享前缀 Token 数；
- Delay、Seed 和可选 Prompt 原文。

默认只保存 SHA-256 Hash，不保存 Prompt 原文；涉及敏感数据时可在部署侧改为密钥化 HMAC。

### 5.3 依赖调度

每个 family 同一时间最多有一个在途请求：

```text
prime 成功 ──> 释放 warm-1 ──> 释放 warm-2
     │
     └─ 失败 ──> 跳过该 family 后续请求
```

不同 family 可以并行执行，从而保留负载能力；family 内部必须保持因果顺序。P0 的单 Runner Delay 限制在 60 秒内，由同一 Worker 内存计时器完成。更长的 TTL 使用已实现的 `cache-retention/v1` Campaign 协议：建图时持久化完整 Dispatch 依赖，Prime 提交后写入 Protocol Instance checkpoint 并解锁到期后继，Planner 只从统一 Dispatch 队列物化 Warm/Cold-Control Runner，不占用 Worker 等待数小时。

### 5.4 分阶段实验

建议按以下顺序推进：

1. 串行存在性测试：并发 1、固定长 Prompt、固定短输出，先确认计数与 TTFT 信号。
2. 前缀长度与突变扫描：改变共享前缀长度、早/晚突变位置，验证信号的单调性。
3. 生产并发与路由测试：逐步提升并发，观察证据是否受排队和实例路由影响。
4. 生命周期测试：改变 prime 到 warm 的间隔，估计外部可见复用窗口。
5. 容量与淘汰测试：插入不同规模的干扰 Prompt，观察命中与延迟衰减。

第 4 阶段已有跨 Runner 独立实例 delay sweep 支持；第 5 阶段的显式干扰流量、容量填充与淘汰模型仍属于后续工作。P0 仍只覆盖同一 Runner、同一 Worker 内的短间隔实验。

## 6. 指标与计算口径

### 6.1 Provider Usage 归一化

当前归一化层识别以下常见结构：

- `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`；
- `prompt_tokens_details.cached_tokens`；
- `input_tokens_details.cached_tokens`；
- `prompt_cache_creation_tokens` / `cache_creation_input_tokens`。

每个请求保存：

- Provider 输入/输出 Token；
- Cache hit/miss/creation Token；
- 来源 Schema、完整性、合法性与校验错误；
- 限长后的原始 Usage 文档。

只有在 Provider 输入 Token 已知且 cached tokens 不超过总输入时，才允许计算：

\[
miss = provider\_input - hit
\]

缺失计数必须保留为 `null`。不得将未知 miss 当成 0，否则只返回 hit 的供应商会被错误计算成 100% 命中。

### 6.2 聚合指标

对于同时具有合法 hit 和 miss 的请求集合：

\[
WeightedHitRatio = \frac{\sum hit_i}{\sum(hit_i + miss_i)}
\]

同时输出：

- 请求总数；
- 包含 hit 计数的请求数；
- 同时包含 hit/miss 的请求数；
- 完整计数覆盖率；
- 请求级命中比例；
- 加权 Token 命中率；
- 非法计数请求数；
- Provider Schema 集合；
- 按 `prime` / `warm` 角色拆分的同口径汇总。

分母未知时结果必须为 `null`，不得用 0 伪造确定性。

### 6.3 配对统计

对每个成功 family 计算：

\[
\Delta TTFT_i = TTFT_{prime,i} - TTFT_{warm,i}
\]

\[
Speedup_i = \frac{TTFT_{prime,i}}{TTFT_{warm,i}}
\]

报告中保存 prime/warm 的中位数与 P95、配对差值的中位数，以及固定 Seed 的 Bootstrap 置信区间。使用配对差值可以消除一部分 Prompt 难度和长度差异，优于把所有 prime 和 warm 分别做非配对均值比较。

### 6.4 流式时序语义

过去的实现把一个携带文本的 SSE Chunk 当成一个 Token，因而旧 `inter_token_latency` 不是严格的 Token 间延迟。一个 Chunk 可能包含零个、一个或多个 Token。

P0 明确区分：

- `time_to_response_headers_s`：请求发出到响应头；
- `time_to_first_sse_s`：请求发出到首个非空 SSE 行；
- `time_to_first_token_s`：请求发出到首个携带文本的 SSE 事件；
- `inter_sse_chunk_latency_s`：相邻携带文本 SSE 事件之间的间隔数组；
- `tpot_s`：若 Provider 返回输出 Token 数，则用首文本到完成的时长除以后续 Token 数；
- `end_to_end_latency_s`：完整请求耗时。

旧 `inter_token_latency` 仅保留用于兼容，并通过 `stream_timing_semantics` 标记为 deprecated。首 Token 延迟不得重复计入解码间隔。

## 7. P0 实施状态

| P0 需求 | 当前实现 | 关键文件 |
|---|---|---|
| Request-plan builder | 不可变 `CacheProbeRequest`；确定性 family 计划；Token 精确突变；Prompt Hash | `src/llmperf/cache_probe.py` |
| Paired workload | `DependentPlanQueue`；prime 成功后释放 warm；失败跳过依赖；family 间并行 | `src/llmperf/cache_probe.py`, `src/llmperf/token_benchmark_ray.py` |
| Complete usage normalization | 从每个 SSE 文档提取 Usage；归一化多种兼容格式；保存 Raw Usage 与校验状态 | `src/llmperf/usage.py`, OpenAI Client |
| Correct aggregation | 缺失值不补零；完整分母、覆盖率、角色拆分和非法计数 | `src/llmperf/cache_analysis.py` |
| Paired statistics | family 内 prime/warm 配对；Delta、Speedup、Bootstrap CI 与 Verdict | `src/llmperf/cache_analysis.py` |
| Timing/provenance | 响应头、首 SSE、首文本、完成时间点；安全响应头；Tokenizer 与请求来源 | OpenAI Client、Backend Tokenizer |
| Tokenizer guard | 显式/绑定/默认选择来源；准确度；解析后的 Revision；近似 Tokenizer 默认拒绝 Cache Probe | Backend Models、Tokenizer Cache、App |
| Streaming timing | 新增 Inter-SSE Chunk 与 TPOT；旧 ITL 保留但标记废弃 | OpenAI Client、Common Metrics |

### 7.1 配置示例

```yaml
benchmark:
  provider: zhipu
  model: glm-5
  llm_api: openai
  tokenizer:
    id: zai-org/GLM-5
    revision: main
    use_fast: true
  concurrent_requests: 1
  mean_input_tokens: 4096
  stddev_input_tokens: 0
  mean_output_tokens: 16
  stddev_output_tokens: 0
  additional_sampling_params:
    temperature: 0
    stream_options:
      include_usage: true
  cache_probe:
    mode: exact_repeat
    trials: 20
    repeats_after_prime: 1
    schedule: randomized_family_blocks
    bootstrap_samples: 2000
    confidence_level: 0.95
    minimum_counter_coverage: 0.8
```

Tokenizer 的 `main` 只是用户请求的 Revision。Backend 接受 Runner 前会把它解析为具体 Commit，并把解析结果写回不可变 Benchmark 配置。若 Cache Probe 最终没有得到不可变 Commit，则拒绝运行。

若完全未指定 Tokenizer，普通性能测试仍使用全局默认 `hf-internal-testing/llama-tokenizer`，结果标记为：

```json
{
  "selection": "global_default",
  "accuracy": "approximate"
}
```

Cache Probe 默认拒绝该近似回退；只有显式设置 `allow_approximate_tokenizer: true` 才能放行，并且 Tokenizer 差异超过阈值时最终 Verdict 会降级为 `inconclusive`。

## 8. 持久化数据设计

### 8.1 是否需要立即增加数据库字段

结论：P0 有必要增加“数据内容”，但没有必要立即增加物理表字段。

当前持久化边界已经具备两个可扩展容器：

- `benchmark_request_results.metrics`：请求级 JSONB；
- `benchmark_runners.summary`：Runner 级 JSONB。

P0 新增的实验字段应先写入这两个容器，以便协议仍在调整时保留兼容性。请求级数据包括：

```text
metrics
├── request_metadata
│   ├── request_id / family_id / role / occurrence
│   ├── plan_index / dispatch_index / completion_index
│   ├── prompt_hash / expected_shared_prefix_tokens
│   └── scheduled_monotonic / dispatched_monotonic
├── normalized_usage
├── raw_usage
├── request_timing
├── response_headers
├── inter_sse_chunk_latency_s
└── latency、token、cache counters
```

Runner Summary 保存：

- Tokenizer 完整来源；
- Cache Probe 参数和计划摘要；
- 全局与按角色 Cache 汇总；
- family 配对统计、置信区间和 Verdict；
- 超时、跳过依赖和 Tokenizer 不一致等质量标志。

这种设计可以重算结果，并避免为每个 Provider 的 Usage 变体频繁迁移 Schema。

### 8.2 建议在 P1 列化的字段

当字段语义稳定、数据量增长或需要 SQL 原生分析时，建议增加以下结构化列：

| 字段 | 建议位置 | 原因 |
|---|---|---|
| `request_id` | request result | 请求幂等、去重和外部关联 |
| `family_id`、`role`、`occurrence` | request result | 高效配对和按角色查询 |
| `plan_index`、`dispatch_index`、`completion_index` | request result | 重建执行顺序、排查并发偏差 |
| `provider_request_id` | request result | 与 Provider 工单和网关日志关联 |
| `dispatched_at`、`completed_at` | request result | 跨进程/跨主机时间关联 |

推荐约束和索引：

```text
UNIQUE (runner_id, request_id)
INDEX  (runner_id, family_id, role, occurrence)
INDEX  (runner_id, dispatch_index)
INDEX  (runner_id, completion_index)
INDEX  (provider_request_id) WHERE provider_request_id IS NOT NULL
```

现有 Monotonic 时间只适合在同一进程内计算耗时，不能作为跨 Worker 的绝对时间。列化时应额外记录 UTC Wall Clock；Duration 计算仍优先使用 Monotonic 时间。

### 8.3 不建议立即列化的数据

以下数据高基数、协议不稳定或体积较大，适合继续保留在 JSONB：

- 完整 Raw Usage；
- Inter-SSE Chunk 延迟数组；
- Provider 私有扩展字段；
- 安全响应头集合；
- Bootstrap 分布细节；
- Prompt 原文。

如果未来需要大规模时序分析，Inter-SSE Chunk 数据应进入独立明细表或列式存储，而不是在主请求表持续增加宽列。

### 8.4 跨 Runner Cache Retention 派发层

分钟级或小时级 TTL 实验不让一个 Worker 常驻等待。Campaign 的 Cache Retention Definition 为每个 `delay × trial` 创建独立 Protocol Instance，并把完整执行图预装载为 Durable Dispatch：Prime 初始为 `pending`，Warm 和可选 Cold-Control 初始为 `blocked`。后继 Dispatch 的 `parent_dispatch_id` 是指向父调用 Dispatch UUID 的自引用外键，并有独立索引；Prompt Hash 仅用于验证 Prime/Warm 载荷一致，不参与因果链寻址。

Planner 不查询或解释 Protocol Instance，只锁定 `pending AND due_at <= now()` 的通用 Dispatch 并物化普通、一次性的 queued Runner。Prime Runner 完成时通过唯一的 `dispatch.runner_id` 找回来源 Dispatch，在同一个 PostgreSQL 事务中把 Prompt Hash 和 UTC 锚点写入实例 checkpoint，再按 `parent_dispatch_id` 把直接子调用从 `blocked` 原子更新为 `pending`。Prime 失败则取消同一组直接子调用。Scheduler 仍然只负责领取 Runner 和监管 Worker。

每个 `delay × trial` 对应一个 `cache-retention/v1` Protocol Instance，避免短 delay 的 Warm 刷新长 delay 样本。实验参数保存在 `spec`，Prompt Hash 与 Prime 锚点保存在 `checkpoint`，实际延迟保存在 `outcome`；Runner ID、阶段状态与到期时间由 Dispatch/Runner 直接提供，不再重复列化。Campaign 生命周期聚合实例状态，因此 Prime 完成但 Warm 尚未到期时仍为 `planned`。

### 8.5 单 Prime 连续驻留协议

`cache-residency/v1` 面向长会话、Agent 周期访问和按小时段观察的工程场景。协议插件
将一个 Prime Runner 编译为多个独立 Prompt 的 bundle，并为每个后续 Warm 预编译稳定
mapping key。运行期只按 key 校验 Prompt Hash，不再推导 Warm 与 Prime 的位置关系。
该结果描述访问条件下的驻留状态，因此不能替代 `cache-retention/v1` 的被动失效曲线。

时间表显式区分两种语义：

- `relative`：观测点是 Prime 实际完成后的秒数；
- `geographic`：从带 UTC offset 且与 IANA timezone 一致的 `starts_at` 开始，按
  `every_seconds` 外推 `duration_days` 个当地日。

地理时间表解决跨天不同业务小时段的派发问题；三天每小时一次会展开为 72 个触发点。
分析横轴仍使用 Warm 开始时间减整个 Prime bundle 实际完成时间。导出同时保存计划
绝对时刻、计划 offset 与实际 delay。Prime
或前置阶段晚于计划点时，子 Dispatch 在依赖完成后作为 overdue work 立即物化，实际
偏差不会被计划时间覆盖。

每个观测点的 Cold Control 使用独立 seed，避免控制请求在后续小时段命中自身缓存；
Warm/Control 顺序按 seed 确定性随机。任一阶段失败会取消仍为 blocked/pending 的后继，
保证 Campaign 不会遗留永远无法释放的调用。

Planner 不占 Runner Slot。多 Planner 通过 PostgreSQL 行锁、原子事务和 `(runner_plan_id, plan_occurrence)` 唯一约束实现幂等。及时到达的 occurrence 即使前一轮仍在执行也会继续入队；等待、停机恢复、Misfire、Overlap 和 DST 规则详见[《LLMPerf Runner Planner 架构与实现》](RUNNER_PLANNER_ARCHITECTURE.zh-CN.md)。

## 9. Scheduler 生命周期与 Cache Probe 边界

当前 Scheduler 随 Backend 生命周期启动并常驻：

1. API 创建 `queued` Runner；
2. Scheduler 事务领取并置为 `running`；
3. 一个槽位创建一个一次性 Worker Ray task/ObjectRef 句柄；
4. Worker task 在 Scheduler 已连接的 shared Ray 中创建请求 Actor 并完成计算；
5. Worker 把 Summary、请求明细和有限日志返回 Scheduler；
6. Scheduler 在事务中持久化结果并释放槽位；
7. Backend 关闭时停止领取、取消受管 Worker task，并由恢复逻辑处理过期 Runner。

P0 的依赖队列存在于 Worker 内部，不改变全局 Scheduler 的 Runner 粒度，适合几十秒内的 prime/warm 实验。长 TTL 由 Prime 完成事务给依赖 Dispatch 写入 `warm_due_at`，再由 Planner 的统一 Dispatch 查询产生一次性 Runner；普通 RunnerPlan 也先编译到同一 Dispatch 协议，只负责日历/周期触发，不承担 Prime/Warm 信息交换。

## 10. Provider 协议兼容策略

当前架构有 OpenAI Chat Completions 客户端，但尚未形成按 Provider 能力主动协商的完整兼容层。建议把兼容性拆成以下层次：

1. Provider Adapter：模型发现、Endpoint、鉴权和错误映射；
2. Protocol Profile：Chat Completions/Responses、SSE 格式和 Usage 开关；
3. Capability Probe：用低成本请求主动验证 stream、usage、cached tokens 和错误语义；
4. Usage Normalizer：把供应商字段投影到统一模型，同时保存 Raw Usage；
5. Tokenizer Binding：Provider/Model 到 Tokenizer 的显式映射与准确度声明。

初步兼容路线：

| Provider | 模型发现 | 推理协议 | Cache Usage | 建议 |
|---|---|---|---|---|
| 智谱 GLM | 不假设存在稳定通用 Models API；静态清单或 Provider 专用发现 | OpenAI 兼容 Chat Completions | 以实际响应能力探测为准 | 401 应透传；避免将“发现失败”包装成无上下文 502 |
| 阿里千问 | DashScope 模型/部署资源与兼容接口并存 | OpenAI 兼容度较高 | 支持时解析 cached token 明细 | 将模型目录与部署 Endpoint 分离 |
| 字节豆包 | 火山方舟 Endpoint/模型资源体系 | OpenAI 兼容需按 Endpoint 能力验证 | 不预设字段存在 | Provider Adapter 负责资源 ID 与模型名转换 |
| 小米 MiMo | 以其公开 OpenAI 兼容入口和模型文档为准 | 优先 Chat Completions 能力探测 | 无字段时依赖配对延迟证据 | 先静态模型配置，再接入专用发现 |

所谓“主动协议归一化”不能靠吞掉错误实现。正确顺序是：读取声明能力、发起最小探测、记录实际响应 Schema、选择 Adapter；401/403、模型不存在和不支持参数必须原样分类返回。

## 11. P1 与 P2 路线

### 11.1 P1：缓存与队列动态

- 前缀长度和突变位置 Sweep；
- 干扰流量、容量压力和淘汰曲线；
- 路由亲和性、实例命中差异与 Header 关联；
- 队列状态代理量和状态空间模型；
- PostgreSQL 上的列化标识、索引和可重算分析任务。

### 11.2 P2：运维与控制

- Provider 能力注册与在线 Capability Probe；
- 多版本统计口径和结果 Schema Version；
- Dashboard、告警与基线漂移检测；
- Cache-aware 负载控制和安全反馈策略；
- 独立列式明细或可观测性后端；
- 数据保留、脱敏、Prompt HMAC 和访问审计。

## 12. 队列状态观测的可行性

服务端排队状态无法由单一 TTFT 直接识别，但可以构建外部拥塞指数。令观测向量为：

\[
y_k = [TTFT_k,\ T_{headers,k},\ T_{first\ sse,k},\ TPOT_k,\ hit_k,\ concurrency_k]
\]

潜在状态可表示为：

\[
x_k = [queue\ delay_k,\ prefill\ load_k,\ decode\ load_k,\ cache\ effectiveness_k]
\]

实际部署应从稳健的滚动分位数和基线残差开始，而不是直接使用复杂滤波器。只有在输入激励充分、观测可识别且能获得真实反馈标签时，才考虑 EKF/UKF、切换模型或粒子滤波器。

缓存会改变 TTFT，但不一定改变排队状态；因此 Cache Hit 必须作为解释变量进入模型，否则观察器可能把缓存收益误判为拥塞下降。

## 13. 验收标准

P0 达标需要满足：

- 相同 Seed 和输入生成相同请求计划与 Prompt Hash；
- warm 不会在 prime 成功前发送；prime 失败时后续依赖可识别为 skipped；
- Usage 可从任意携带它的 SSE 文档提取，包括同时携带文本的事件；
- 缺少 miss 时命中率返回 `null`，非法计数不参与有效聚合；
- 配对分析输出样本数、Delta、Speedup、Bootstrap 区间和 Verdict；
- 记录响应头、首 SSE、首文本、完成时间与安全 Request ID；
- 默认 Tokenizer 的来源和近似性质可见，受控 Cache Probe 要求显式 Tokenizer；
- Inter-SSE Chunk 与 TPOT 语义明确，旧 ITL 标记废弃；
- 原始请求数据足以重算 Summary；
- 失败异常能从 Ray/线程传播到 Worker、Scheduler、Runner 和 CLI。

当前验证重点为纯单元和请求客户端集成模拟；按需求不执行 SQLite 数据库测试。生产数据库字段与索引应在 PostgreSQL 迁移和集成环境中单独验收。

## 14. 风险与限制

- Provider 的 OpenAI 兼容性可能只覆盖请求格式，不覆盖 Usage、SSE 或错误结构；
- Tokenizer 即使来自同一模型仓库，也可能与服务端私有预处理存在差异；
- 网关重试、负载均衡和实例迁移会破坏 family 内缓存亲和性；
- Provider 可能异步构建缓存，prime 返回成功不代表缓存已立即可见；
- Bootstrap 区间反映当前样本分布，不消除系统性偏差；
- Raw Usage 和响应头仍需限长、白名单和数据保留策略；
- P0 在 Runner 完成后批量写结果，运行中尚不可查询逐请求部分结果。

## 15. 参考资料

### 推理与缓存系统

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, 2023.
- vLLM Documentation, Automatic Prefix Caching.
- SGLang Documentation, RadixAttention and Prefix Caching.

### 统计与观测

- Efron and Tibshirani, *An Introduction to the Bootstrap*.
- Jain, *The Art of Computer Systems Performance Analysis*.
- Kalman, *A New Approach to Linear Filtering and Prediction Problems*.

### Provider 文档

- 智谱开放平台：模型与 Chat Completions API 文档。
- 阿里云百炼/DashScope：OpenAI 兼容接口、模型与缓存计量文档。
- 火山方舟：模型调用、Endpoint 与 OpenAI 兼容文档。
- 小米 MiMo：模型与 API 使用文档。

Provider 文档和线上协议会变化，正式兼容性矩阵应以受测日期、Endpoint 和 Capability Probe 结果为准。
