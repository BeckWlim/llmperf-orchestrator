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

Campaign 文件必须包含 `campaign`，并至少包含非空的 `runners` 或
`runner_plans`。两者可以同时存在：

```yaml
campaign:
  name: deepseek-study
  description: bounded benchmark campaign
  tags:
    provider: aliyun
wait: true
runners: []
runner_plans: []
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
- `mean_output_tokens`、`stddev_output_tokens`：输出 token 分布。
- `shared_prefix_tokens`：普通共享前缀长度；CacheProbe 也可单独声明。
- `additional_sampling_params`：原样加入兼容推理请求，例如 `temperature: 0`
  和 `stream_options.include_usage: true`。
- `dataset_repeat_count`、`dataset_seed`：数据集重复与可复现实验顺序。

所有配置模型使用严格字段校验；拼错或未知字段会被拒绝，不要依赖静默忽略。

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

## 6. 修改前校验

1. 对照 `src/llmperf_backend/models.py` 确认字段和范围。
2. 对照 `examples/runner-plan.yaml`、`examples/test-smoke.yaml` 和
   `examples/test-campaign.yaml`。
3. 用 `llmperfctl planner preview -f FILE` 检查 RunnerPlan occurrence。
4. 先用 1x1 smoke Runner 验证 provider/model/key，再运行长 Campaign。
