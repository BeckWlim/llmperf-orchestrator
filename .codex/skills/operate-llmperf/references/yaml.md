# LLMPerf YAML 规范

## 1. 选择顶层形态

单次 Runner 文件只包含 `label`、`metadata`、`benchmark`：

```yaml
label: deepseek-smoke-1x1
metadata:
  purpose: verify-end-to-end
benchmark:
  provider: aliyun
  model: deepseek-v4-pro
  timeout_seconds: 30
  max_completed_requests: 1
  concurrent_requests: 1
  mean_input_tokens: 64
  stddev_input_tokens: 0
  mean_output_tokens: 16
  stddev_output_tokens: 0
  additional_sampling_params:
    temperature: 0
```

Campaign 文件必须包含 `campaign`，并至少包含非空的 `runners`、
`runner_plans` 或 `protocol_definitions`。三者可以同时存在：

```yaml
campaign:
  name: deepseek-study
  description: bounded benchmark campaign
  tags:
    provider: aliyun
wait: true
runners: []
runner_plans: []
protocol_definitions: []
```

不要在 Runner YAML 中放 endpoint 或 API key。`provider` 是 Backend Provider
Profile ID，`model` 应使用该 Profile 可见的精确模型 ID。

## 2. Benchmark 字段

常用字段及作用：

- `timeout_seconds`：每个模型 HTTP 请求的上限。
- `max_completed_requests`：Runner 目标完成请求数。
- `concurrent_requests`：Runner 内同时在途的请求数；增大它改变并发压力，
  不代表增加 Runner 或 Scheduler slot。
- `mean_input_tokens`、`stddev_input_tokens`：输入 token 分布。
- `mean_output_tokens`、`stddev_output_tokens`：每次请求 `max_tokens` 上限的目标分布；
  模型仍可能因为 EOS、过滤或 Provider 行为提前停止，不能把它解释为实际输出保证。
- `shared_prefix_tokens`：普通共享前缀长度；CacheProbe 也可单独声明。
- `additional_sampling_params`：原样加入兼容推理请求，例如 `temperature: 0`
  和 `stream_options.include_usage: true`。
- `dataset_repeat_count`、`dataset_seed`：数据集重复与可复现实验顺序。

所有配置模型使用严格字段校验；拼错或未知字段会被拒绝，不要依赖静默忽略。

提交前估算计划 Provider 请求数，并向用户说明较大的预算：

- 普通 Runner：`max_completed_requests`。
- 多 Runner Campaign：所有即时 Runner 请求数之和。
- RunnerPlan：每轮请求数乘以最大 occurrence 数；只有 `ends_at` 时按 preview 结果计算。
- `cache_probe`：`trials × (1 + repeats_after_prime)`。
- `cache-retention/v1`：`delay 数 × trials_per_delay × (2 + cold_control)`。
- `cache-residency/v1`：`chains × observation 数 × (2 + cold_control)`；Prime bundle
  虽是一个 Runner，但包含每个 observation 的独立 Prime 请求。

其中 `cold_control` 为 true 时在公式中取 1，否则取 0。超时、失败和人工重跑可能令实际
计费调用与计划数不同，不要把估算当作账单保证。

## 3. Tokenizer 与 Dataset

未声明 tokenizer 时会使用 Backend 全局默认 tokenizer，其选择标记为
`global_default`，精度为 `approximate`。一般吞吐测试可以接受；KV Cache 精确探针
应显式声明：

```yaml
tokenizer:
  source: huggingface
  id: deepseek-ai/DeepSeek-V3
  revision: main
  use_fast: true
```

提交时 Backend 会把可变 revision 解析成 Hugging Face immutable commit，并持久化
解析结果。下载成功不等于对象已经携带 commit hash；诊断时检查最终解析日志中的
`resolved revision` 和 `immutable`。

Hugging Face ShareGPT 数据集声明：

```yaml
dataset:
  source: huggingface
  id: anon8231489123/ShareGPT_Vicuna_unfiltered
  filename: ShareGPT_V3_unfiltered_cleaned_split.json
  revision: main
  format: sharegpt
dataset_repeat_count: 4
dataset_seed: 22222
```

Tokenizer/Dataset 由 Backend 下载并缓存。代理、离线模式及缓存目录属于 Backend
环境配置，不写进任务 YAML。

## 4. KV Cache Probe

显式 tokenizer 后才启用精确探针：

```yaml
cache_probe:
  mode: exact_repeat
  trials: 20
  repeats_after_prime: 1
  delay_seconds: 0
  bootstrap_samples: 2000
  confidence_level: 0.95
  minimum_counter_coverage: 0.8
```

支持 `exact_repeat`、`shared_prefix`、`early_mutation`、`late_mutation`。
`shared_prefix` 必须提供 `shared_prefix_tokens`，且小于 `mean_input_tokens`。
只有明确接受误差时才设置 `allow_approximate_tokenizer: true`。不要把未知 cache
counter 当作 0；覆盖率不足时相关聚合结果应为 `null` 或不可判定。

## 5. RunnerPlan 时间配置

推荐的 30 秒间隔、8 轮、保留所有轮次配置：

```yaml
campaign:
  name: deepseek-v4-pro-cache-campaign
runner_plans:
  - name: deepseek-v4-pro-cache-30s-8x
    timezone: Asia/Shanghai
    max_occurrences: 8
    recurrence:
      kind: interval
      every_seconds: 30
    overlap_policy: queue
    misfire_grace_seconds: 300
    runner:
      label: deepseek-v4-pro-cache-30s
      benchmark:
        provider: aliyun
        model: deepseek-v4-pro
        tokenizer:
          id: deepseek-ai/DeepSeek-V3
          revision: main
          use_fast: true
        concurrent_requests: 1
        mean_input_tokens: 4096
        stddev_input_tokens: 0
        mean_output_tokens: 16
        stddev_output_tokens: 0
        cache_probe:
          mode: exact_repeat
          trials: 20
```

时间规则：

- `timezone` 必须是 IANA 时区。
- 必须提供 `ends_at` 或 `max_occurrences`，禁止无边界永久计划。
- `starts_at` 如提供，必须带 UTC offset；缺省时以 PostgreSQL 事务时间冻结为
  起点，第 0 轮立即到期。
- 缺省 `starts_at` 时 `misfire_grace_seconds` 必须大于 0；为首次派发预留宽容，
  推荐短周期测试使用 300 秒。
- `overlap_policy: queue` 始终把及时 occurrence 写入普通 Runner 队列。
- `overlap_policy: skip` 在前一 Runner 仍为 queued/running 时跳过本轮并记审计。
- `interval` 使用 `every_seconds`。地理日历使用 `kind: calendar`、
  `frequency: daily|weekly`、`local_time`，weekly 还必须提供 `weekdays`。

以 Campaign YAML 启动周期工作负载。`planner create` 仅用于把一个计划附加到已经
存在的 Campaign；不要把 Planner 当作独立工作负载所有者。

## 6. 跨 Runner Cache Retention Sweep

长 TTL 不使用 `cache_probe.delay_seconds` 让 Worker 常驻等待，而是在 Campaign 中
声明持久化、独立 family 的 delay sweep：

```yaml
protocol_definitions:
  - name: deepseek-cache-retention
    protocol: cache-retention/v1
    delay_seconds: [0, 30, 300, 1800]
    trials_per_delay: 20
    seed: 11111
    assignment: randomized_blocks
    refresh_semantics: independent_family
    cold_control: true
    runner:
      benchmark:
        provider: aliyun
        model: deepseek-v4-pro
        tokenizer:
          id: deepseek-ai/DeepSeek-V3
          revision: main
        mean_input_tokens: 4096
        stddev_input_tokens: 0
        mean_output_tokens: 16
        stddev_output_tokens: 0
        additional_sampling_params:
          stream_options:
            include_usage: true
```

每个 `delay × trial` 使用独立 Prompt family。建图时 Prime Dispatch 为 `pending`，
Warm 与可选 Cold Control Dispatch 为 `blocked`，并通过 Prime Dispatch UUID 建立依赖。
Prime 完成事务把直接后继解锁并将到期时间设为 `prime_completed_at + delay`；Planner
只物化统一协议中已到期的 `pending` Dispatch。等待期间不占 Scheduler slot。当前协议
要求 OpenAI 客户端、显式或模型绑定的 immutable tokenizer，以及固定输入/输出长度。
Campaign export v5 保存通用 Protocol Definition/Instance、Dispatch 调用图和按 delay
聚合的缓存命中/TTFT 协议分析；不再暴露 Cache Pair 专用持久化模型。

## 7. 单 Prime 长时间驻留链

`cache-residency/v1` 用一个 Prime Runner 打包多个独立 Prompt，并把后续 Warm 在
YAML 编译阶段一对一映射到这些 Prompt。它测量的是访问条件下的缓存驻留，不得解释为
无中间访问的自然 TTL。时间表有两种显式语义：

```yaml
protocol_definitions:
  - name: deepseek-cache-residency
    protocol: cache-residency/v1
    schedule:
      kind: relative
      offsets_seconds: [3600, 7200, 14400]
    mapping: one_to_one
    chains: 1
    seed: 22222
    cold_control: true
    runner:
      benchmark:
        provider: aliyun
        model: deepseek-v4-pro
        tokenizer:
          id: deepseek-ai/DeepSeek-V3
          revision: main
        mean_input_tokens: 4096
        stddev_input_tokens: 0
        mean_output_tokens: 16
        stddev_output_tokens: 0
```

`relative` 以整个 Prime bundle 实际完成时间为零点。需要跨多个本地日按小时观测时，
使用有界的地理周期时间表：

```yaml
schedule:
  kind: geographic
  timezone: Asia/Shanghai
  starts_at: 2026-08-15T00:00:00+08:00
  every_seconds: 3600
  duration_days: 3
```

`starts_at` 必须携带 UTC offset，且 offset 必须与声明的 IANA 时区在该时刻一致。
插件按当地日边界外推时间表；上述配置产生 72 个逐小时观测点。`starts_at` 决定 Prime
bundle 的绝对派发时刻，派生触发点由周期展开；报告仍以 Prime bundle 实际完成时间
计算每次 Warm 的真实缓存年龄。Prime 延迟导致某个观测点已经过期时，该阶段在前置
依赖完成后立即进入通用到期队列，并在结果中体现真实延迟。

`mapping: one_to_one` 令插件为每个观测点生成一个稳定 `mapping_key`；Prime bundle
请求与 Warm Dispatch 直接携带该 key，运行时只校验 `mapping_key -> prompt_hash`，不再
推导数组索引。每个 chain 编译为严格因果链。Warm/Control 的先后顺序按 seed 确定性
随机化，每个 Cold Control 使用不同 Prompt seed。任一阶段失败会取消尚未派发的后继；
等待期间不占 Worker。`chains` 用于创建独立复现链，默认为 1。

## 8. 修改前校验

1. 对照 `src/llmperf_backend/models.py` 确认字段和范围。
2. 对照 `examples/example-runner-plan.yaml`、`examples/example-smoke.yaml` 和
   `examples/example-campaign.yaml`。
3. 用 `.venv/bin/python .codex/skills/operate-llmperf/scripts/validate_workload.py FILE`
   执行无 Provider 调用的本地严格校验。
4. 用 `llmperfctl planner preview -f FILE` 检查 RunnerPlan occurrence。
5. 先用 1x1 smoke Runner 验证 provider/model/key，再运行长 Campaign。
