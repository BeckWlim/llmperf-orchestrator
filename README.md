# LLMPerf

A Tool for evaulation the performance of LLM APIs.

# Installation
```bash
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

Ray is part of the benchmark execution path: request clients run as Ray Actors.
The project therefore installs `ray[default]`, not the minimal `ray` package;
the default extra supplies the dashboard, event, and metrics-agent dependencies
used during `ray.init()`. If an existing environment was created with minimal
Ray, refresh it with `python -m pip install -e .` before running benchmarks.

## Configuration Backend

LLMPerf includes an optional FastAPI backend for validating and reloading YAML
runtime configuration. The default configuration is packaged at
`src/llmperf_backend/configs/default.yaml`.

Start the backend after installation:

```bash
cp .env.template .env
# Edit DATABASE_URL, the model endpoint, and credentials in .env.
llmperf-backend
```

The backend automatically loads `.env` from its current working directory.
Exported process variables take precedence. Set `LLMPERF_ENV_FILE` before
startup to select another dotenv file; changing dotenv values requires a
service restart. Workers and Ray actors inherit the resolved provider
credentials, while task payloads and JSON exports never contain those secrets.

`.env.template` intentionally contains only PostgreSQL, provider credentials,
default provider/model, and a few optional service controls. Benchmark workload
parameters belong in submitted YAML. The real `.env`, `.env.*`, and `.secrets/`
are ignored by Git; `.env.template` is the only allowed dotenv template.

Each submitted Runner can select its own tokenizer. The backend resolves and caches
the tokenizer before accepting the Runner, records the resolved revision with the
immutable benchmark configuration, and gives the Worker only a local directory:

```yaml
benchmark:
  provider: aliyun
  model: glm-5.2
  tokenizer:
    id: THUDM/glm-4-9b-chat
    revision: main
    use_fast: true
```

`source` defaults to `huggingface`. Remote tokenizer code is never trusted or
executed. Cache location and offline-only lookup are controlled by the backend:

```dotenv
LLMPERF_TOKENIZER_CACHE_DIR=/var/cache/llmperf/tokenizers
LLMPERF_TOKENIZER_PROXY=http://proxy.example.com:3128
LLMPERF_TOKENIZER_LOCAL_FILES_ONLY=false
```

`LLMPERF_TOKENIZER_PROXY` is shared by tokenizer and dataset resolution and is
passed directly to their HTTP and HTTPS Hugging Face requests. It is useful when
the backend service cannot see a desktop or system proxy. Standard `HTTP_PROXY`,
`HTTPS_PROXY`, and `NO_PROXY` environment variables remain available for
process-wide networking.

Workers always pass `local_files_only=True`, fail immediately when their resolved
directory is invalid, and cache one tokenizer instance per process. Runner YAML
that omits `tokenizer` uses the backend default
`hf-internal-testing/llama-tokenizer`, resolved through the same cache.

To use another configuration file, set its path before starting the server:

```bash
export LLMPERF_BACKEND_CONFIG=/path/to/backend.yaml
llmperf-backend
```

The backend exposes:

- `GET /health`
- `GET /api/v1/scheduler/status`
- `GET /api/v1/providers`
- `GET /api/v1/providers/{provider_id}/models`
- `GET /api/v1/config`
- `GET /api/v1/config/schema`
- `POST /api/v1/config/validate`
- `POST /api/v1/config/reload`
- `POST/GET /api/v1/campaigns`
- `GET /api/v1/campaigns/{campaign_id}`
- `POST /api/v1/campaigns/{campaign_id}/cancel`
- `POST /api/v1/campaigns/{campaign_id}/runners`
- `GET /api/v1/campaigns/{campaign_id}/export`
- `POST/GET /api/v1/runners`
- `POST /api/v1/runners/{runner_id}/cancel`
- `GET /api/v1/runners/{runner_id}/results`
- `GET /api/v1/runners/{runner_id}/export`

YAML environment placeholders use `${NAME}` for required values or
`${NAME:-default}` for optional values. Configuration files are parsed with
`yaml.safe_load`; keep API keys and other secrets in environment variables.

The backend also provides PostgreSQL-backed asynchronous task orchestration.
Results are committed to the database first and JSON is generated only through
the export endpoints. See [the architecture guide](docs/ARCHITECTURE.md) for the
data model, task state machine, API, and lightweight `llmperfctl` workflow.

Provider credentials are backend-owned profiles. A task selects only a profile
and a model, so its YAML does not carry an API key or endpoint:

```dotenv
LLMPERF_PROVIDER_DEEPSEEK_URL=https://api.deepseek.com/v1
LLMPERF_PROVIDER_DEEPSEEK_KEY=replace-with-real-key
LLMPERF_DEFAULT_PROVIDER=deepseek
LLMPERF_DEFAULT_MODEL=deepseek-chat
```

For an OpenAI-compatible endpoint, `ADAPTER=openai`, model discovery through
`/models`, and a 300-second cache are inferred. At least one Provider Profile is
required; legacy `API_BASE`, `API_KEY`, `LLM_API`, and implicit `default`
Provider fields and implicit `default` profiles are not supported.

```yaml
label: deepseek-smoke
benchmark:
  provider: deepseek
  model: deepseek-chat
  timeout_seconds: 30
  max_completed_requests: 1
  concurrent_requests: 1
  mean_input_tokens: 64
  stddev_input_tokens: 0
  mean_output_tokens: 16
  stddev_output_tokens: 0
```

The backend derives `llm_api`, endpoint, and key from the selected profile.
Inspect configured profiles and discover the models visible to a profile key
through the CLI:

```bash
llmperfctl provider list
llmperfctl provider models deepseek
llmperfctl provider models deepseek --refresh
```

Discovery uses the provider's configured `/models` endpoint when compatible,
or an administrator-maintained static model list. It establishes catalog
visibility for the key, not that a completion request will succeed.

For example, upload and orchestrate a complete GLM campaign with:

```bash
llmperfctl campaign start -f examples/glm-campaign.yaml --wait
```

`campaign start` validates every Runner and creates the Campaign, Runners, and
initial Runner events in one database transaction. A rejected Runner batch does
not leave an empty Campaign behind. Empty Campaigns created by older two-request
CLI versions remain valid historical records but can contain no results.

To measure provider-reported KV-cache reuse with the standard ShareGPT serving
dataset, select a persistent backend cache and run the DeepSeek campaign:

```bash
# Set this in the backend service environment (or its .env), then restart it:
LLMPERF_DATASET_CACHE_DIR=/var/cache/llmperf/datasets

# Run the client after the backend is ready:
llmperfctl campaign start \
  -f examples/deepseek-v4-pro-kvcache-campaign.yaml --full
```

The CLI submits the Hugging Face dataset specification from the Campaign and
receives queued Runners immediately. The Scheduler downloads the declared
artifact once, stores it under `LLMPERF_DATASET_CACHE_DIR` (default
`~/.cache/llmperf/datasets`), and gives Workers only its resolved local path.
Large downloads therefore cannot time out the Runner-submission HTTP request;
download failures become durable failed Runner outcomes. Hugging Face's standard
environment variables control authentication, proxies, and offline operation.

The unique control issues eight different first-turn prompts. The repeated
workload samples two different prompts and issues each four times, matching the
repeat strategy used by vLLM's automatic-prefix-caching benchmark. Look for
`summary.results.kv_cache.hit_ratio` in each completed Runner. Input prompts are
filtered to 1,024-5,120 tokens; change `mean_input_tokens` and
`stddev_input_tokens` together to select another range.

The public control model has four distinct responsibilities:

- `Scheduler`: backend-owned queue consumer; query with
  `llmperfctl scheduler status`.
- `Runner`: one durable benchmark execution; manage with
  `runner start/status/list/wait/cancel/logs/export`.
- `Worker`: temporary subprocess for one Runner attempt; its PID, exit code,
  stdout, and stderr are reported through Runner status/logs.
- `Campaign`: durable grouping of Runners; manage with
  `campaign start/status/list/cancel/export`.

Schedulers and Workers follow the backend lifecycle and cannot be started
directly through the remote CLI.

`runner start` is non-blocking by default. It prints submission logs to stderr,
prints the accepted Runner (including `runner_id` and `status`) to stdout, and
then exits so the ID can be used later:

```bash
llmperfctl runner start -f examples/glm-smoke.yaml
llmperfctl runner status RUNNER_ID
```

Wait explicitly with `-w` or `--wait`:

```bash
llmperfctl runner start -f examples/glm-smoke.yaml -w
llmperfctl runner status RUNNER_ID -w
```

Runner listings use a compact table and return at most 20 rows by default:

```bash
llmperfctl runner list
llmperfctl runner list --status failed --limit 10
llmperfctl runner list --json   # lightweight machine-readable records
llmperfctl runner list --full   # complete Runners, summaries, stdout and stderr
```

The default list API is also a lightweight projection, so large benchmark
summaries and captured logs are not transferred only to be hidden by the CLI.
Use `full=true` (or CLI `--full`) only for diagnostics.
The CLI and backend use one strict list-response contract.

`runner wait` and `runner start --wait` print a compact outcome by default. Add
`--full` only when the complete Runner document and captured stdout/stderr are
needed. Status transitions (`queued`, `running`, terminal status) are printed as
timestamped logs on stderr as they occur; JSON/table results remain on stdout.
Set `--log-level debug` or `LLMPERFCTL_LOG_LEVEL=debug` for more CLI detail.
Interactive terminals highlight log levels by default. Use `--color always` when
redirecting to a color-capable viewer, `--color never` for plain output, or
`LLMPERFCTL_LOG_COLOR` as the CLI default. Backend and Worker logs use
`LLMPERF_LOG_COLOR=auto|always|never`.
`runner status RUNNER_ID --summary` provides the same compact view, and adding
`--wait` reconnects to an already submitted Runner until it becomes terminal.
A failed or cancelled waited Runner exits with code 2 for shell/CI use.
`--timeout` limits only local CLI waiting and does not cancel the durable Runner.

Campaign status is aggregate by default. Request the complete JSON report,
including Runner summaries and captured logs, with `--full`; add
`--include-requests` for individual request metrics:

```bash
llmperfctl campaign status CAMPAIGN_ID --full
llmperfctl campaign status CAMPAIGN_ID --full --include-requests
llmperfctl campaign export CAMPAIGN_ID -o campaign-report.json
```

A Worker process exiting normally is not sufficient for benchmark success.
Runners with zero completed model requests are stored as `failed` together with
their summary and request errors; partially successful runs remain `succeeded`
with `summary.outcome.status=degraded`. Failed runs with persisted results can
still be exported for diagnosis.

The benchmark `timeout_seconds` also bounds each OpenAI-compatible HTTP request;
the client no longer uses an unrelated fixed 180-second timeout. This prevents
a stalled provider request from making a short smoke run appear hung.

When neither `--token` nor `--private-key` is supplied, `llmperfctl` discovers
secure, unencrypted RSA private keys in `~/.ssh` and retries the request with the
next key only after an HTTP 401 response. It prioritizes `~/.ssh/llmperfctl` and
`~/.ssh/id_rsa`. Use `--ssh-dir`, `LLMPERF_SSH_DIR`, or
`--no-key-discovery` to control this behavior. Explicit credentials always win.

The API can trust only designated CLI instances using a fixed PEM public key.
The service verifies short-lived RS256 tokens while `llmperfctl` retains the
private key and refreshes tokens automatically. Setup and key-generation steps
are documented in [the architecture guide](docs/ARCHITECTURE.md#固定公钥认证配置).

Initialize a PostgreSQL database explicitly with:

```bash
psql -v ON_ERROR_STOP=1 -d llmperf -f sql/postgresql/init.sql
```

The SQL creates task, metric, user, trusted-key, and audit tables but does not
seed a database superuser. Bootstrap public-key authentication handles the
first trusted CLI call. Real PostgreSQL repository tests are opt-in through
`LLMPERF_TEST_DATABASE_URL`; details are in the architecture guide.

# Basic Usage

We implement 2 tests for evaluating LLMs: a load test to check for performance and a correctness test to check for correctness.

## Load test

The load test spawns a number of concurrent requests to the LLM API and measures the inter-token latency and generation throughput per request and across concurrent requests. The prompt that is sent with each request is of the format:

```
Randomly stream lines from the following text. Don't generate eos tokens:
LINE 1,
LINE 2,
LINE 3,
...
```

Where the lines are randomly sampled from a collection of lines from Shakespeare sonnets. Tokens are counted using the `LlamaTokenizer` regardless of which LLM API is being tested. This is to ensure that the prompts are consistent across different LLM APIs.

To run the most basic load test you can the token_benchmark_ray script.


### Caveats and Disclaimers

- The endpoints provider backend might vary widely, so this is not a reflection on how the software runs on a particular hardware.
- The results may vary with time of day.
- The results may vary with the load.
- The results may not correlate with users’ workloads.

### OpenAI Compatible APIs
```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api openai \
--additional-sampling-params '{}'

```

### Anthropic
```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python token_benchmark_ray.py \
--model "claude-2" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api anthropic \
--additional-sampling-params '{}'

```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python token_benchmark_ray.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python token_benchmark_ray.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### LiteLLM

LLMPerf can use LiteLLM to send prompts to LLM APIs. To see the environment variables to set for the provider and arguments that one should set for model and additional-sampling-params.

see the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

```bash
python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Vertex AI

Here, --model is used for logging, not for selecting the model. The model is specified in the Vertex AI Endpoint ID.

The GCLOUD_ACCESS_TOKEN needs to be somewhat regularly set, as the token generated by `gcloud auth print-access-token` expires after 15 minutes or so.

Vertex AI doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "vertexai" \
--additional-sampling-params '{}'

```

### SageMaker

SageMaker doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

see `python token_benchmark_ray.py --help` for more details on the arguments.

## Correctness Test

The correctness test spawns a number of concurrent requests to the LLM API with the following format:

```
Convert the following sequence of words into a number: {random_number_in_word_format}. Output just your final answer.
```

where random_number_in_word_format could be for example "one hundred and twenty three". The test then checks that the response contains that number in digit format which in this case would be 123.

The test does this for a number of randomly generated numbers and reports the number of responses that contain a mismatch.

To run the most basic correctness test you can run the the llm_correctness.py script.

### OpenAI Compatible APIs

```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE=https://console.endpoints.anyscale.com/m/v1

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--max-num-completed-requests 150 \
--timeout 600 \
--num-concurrent-requests 10 \
--results-dir "result_outputs"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python llm_correctness.py \
--model "claude-2" \
--llm-api "anthropic"  \
--max-num-completed-requests 5 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs"
```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python llm_correctness.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python llm_correctness.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### LiteLLM

LLMPerf can use LiteLLM to send prompts to LLM APIs. To see the environment variables to set for the provider and arguments that one should set for model and additional-sampling-params.

see the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

```bash
python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

see `python llm_correctness.py --help` for more details on the arguments.


### Vertex AI

Here, --model is used for logging, not for selecting the model. The model is specified in the Vertex AI Endpoint ID.

The GCLOUD_ACCESS_TOKEN needs to be somewhat regularly set, as the token generated by `gcloud auth print-access-token` expires after 15 minutes or so.

Vertex AI doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.


```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "vertexai" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### SageMaker

SageMaker doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

## Saving Results

The results of the load test and correctness test are saved in the results directory specified by the `--results-dir` argument. The results are saved in 2 files, one with the summary metrics of the test, and one with metrics from each individual request that is returned.

# Advanced Usage

The correctness tests were implemented with the following workflow in mind:

```python
import ray

from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import get_tokenizer


# Copying the environment variables and passing them to ray.init() is necessary
# For making any clients work.
ray.init(runtime_env={"env_vars": {"OPENAI_API_BASE" : "https://api.endpoints.anyscale.com/v1",
                                   "OPENAI_API_KEY" : "YOUR_API_KEY"}})

base_prompt = "hello_world"
tokenizer = get_tokenizer()
base_prompt_len = len(tokenizer.encode(base_prompt))
prompt = (base_prompt, base_prompt_len)

# Create a client for spawning requests
clients = [OpenAIChatCompletionsClient.remote()]

req_launcher = RequestsLauncher(clients)

req_config = RequestConfig(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt=prompt
    )

req_launcher.launch_requests(req_config)
result = req_launcher.get_next_ready(block=True)
print(result)

```

# Implementing New LLM Clients

To implement a new LLM client, you need to implement the base class `llmperf.ray_llm_client.LLMClient` and decorate it as a ray actor.

```python

from llmperf.ray_llm_client import LLMClient
import ray


@ray.remote
class CustomLLMClient(LLMClient):

    def llm_request(self, request_config: RequestConfig) -> Tuple[Metrics, str, RequestConfig]:
        """Make a single completion request to a LLM API

        Returns:
            Metrics about the performance charateristics of the request.
            The text generated by the request to the LLM API.
            The request_config used to make the request. This is mainly for logging purposes.

        """
        ...

```

# Legacy Codebase
The old LLMPerf code base can be found in the [llmperf-legacy](https://github.com/ray-project/llmval-legacy) repo.
