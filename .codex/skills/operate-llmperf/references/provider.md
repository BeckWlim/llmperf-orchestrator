# Provider Profile 配置与验证

## 目录

1. 配置边界
2. 检查现有 Profile
3. 创建 OpenAI-compatible Profile
4. 其他 Adapter 与模型发现
5. 配置加载与 Worker 注入
6. 验证与排障

## 1. 配置边界

Provider Profile 由 Backend 拥有。Runner/Campaign YAML 只写稳定的 `provider` ID 和
精确 `model` ID，不写 endpoint、API key，也不要依赖 Worker 进程的桌面 shell 环境。

只有用户要求配置 Provider、或已要求运行但目标 Profile 不存在/不可用时才修改配置。
缺少密钥时不要猜测或要求用户把密钥贴进对话；给出 `--stdin` 命令并让用户安全输入。

## 2. 检查现有 Profile

先做只读检查：

```bash
llmperfctl health
llmperfctl scheduler status
llmperfctl provider list
llmperfctl provider models PROVIDER_ID
```

`provider list` 只公开 `id`、adapter、base URL、`has_api_key`、发现模式等安全字段。
`provider models --refresh` 绕过内存 TTL 缓存并访问远端，需要 operator 权限；普通检查
优先不加 `--refresh`。模型目录可见只证明 catalog API 可用，不证明 inference 可用。

## 3. 创建 OpenAI-compatible Profile

Profile 环境变量形式是 `LLMPERF_PROVIDER_<ID>_<FIELD>`。ID 使用大写下划线书写时会
归一化成小写连字符，例如 `TEAM_OPENAI` 对应 YAML 中的 `team-openai`。

```bash
llmperf-backend config set LLMPERF_PROVIDER_ACME_URL https://api.example.com/v1
llmperf-backend config set LLMPERF_PROVIDER_ACME_ADAPTER openai
printf '%s' "$ACME_API_KEY" | \
  llmperf-backend config set LLMPERF_PROVIDER_ACME_KEY --stdin
llmperf-backend config set LLMPERF_PROVIDER_ACME_DISCOVERY openai
llmperf-backend config set LLMPERF_DEFAULT_PROVIDER acme
llmperf-backend config list
```

不要把真实 key 放在命令参数中。URL 必须是 HTTP(S)，且不能包含 userinfo、query 或
fragment；Backend 会去掉末尾 `/`。默认模型发现路径为 `/models`。

`LLMPERF_DEFAULT_PROVIDER` 只影响未显式选择 Provider 的默认 Benchmark；checked-in
workload 应显式写 `provider`，避免环境变化改变实验目标。

## 4. 其他 Adapter 与模型发现

支持的 adapter 是 `openai`、`anthropic`、`litellm`、`sagemaker`、`vertexai`。
Profile 的 adapter 会覆盖 workload 中的 `llm_api`，因此不要用 YAML 绕过 Profile。

远端没有 OpenAI `/models` 时使用静态目录：

```bash
llmperf-backend config set LLMPERF_PROVIDER_ACME_DISCOVERY static
llmperf-backend config set LLMPERF_PROVIDER_ACME_MODELS model-a,model-b
```

`DISCOVERY` 取 `openai|static|disabled`。`static` 必须配置 `MODELS`；`disabled` 允许已知
精确模型用于运行，但 `provider models` 会失败。`PATH` 可覆盖发现路径，`TTL` 控制模型
缓存秒数且范围为 0–86400。仅在 adapter 客户端需要非默认变量名时设置 `URLVAR`、
`KEYVAR`。使用 `litellm`、SageMaker 等可选集成前先确认对应依赖已安装。

## 5. 配置加载与 Worker 注入

Backend 配置优先级为：进程环境、`LLMPERF_ENV_FILE`、用户持久化 backend config、当前
目录 `.env`。`llmperf-backend config set` 修改用户持久化文件，不热更新正在运行的
ProviderRegistry；任何 Profile、默认 Provider、URL 或 key 变化后都必须重启 Backend。

Scheduler 启动 Worker 时会移除所有 Profile 的 endpoint/key 变量，只把当前 Runner
选中的 Profile 注入 Worker。不要把 secret 复制到 YAML、metadata、日志或导出文件。

## 6. 验证与排障

重启后按顺序验证：

1. `llmperfctl health` 与 `scheduler status`。
2. `provider list` 确认 ID、adapter、URL 和 `has_api_key`。
3. `provider models ID` 确认精确 model ID；必要时才用 `--refresh`。
4. 使用 1 并发、1 请求、短 token/timeout 的 smoke Runner 证明 inference。
5. smoke 失败时执行 `runner status ID --summary`，再执行 `runner logs ID`。

401 通常是 Provider key 无效或未注入；404/unknown model 通常是 model ID 或 API base
路径不匹配；catalog 成功但 inference 失败时以 smoke 的 HTTP 错误为准。Provider 配置
改变后若行为未变化，先确认 Backend 确实已重启且加载的是预期 config path。
