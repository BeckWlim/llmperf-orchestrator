# LLMPerf 部署操作参考

## 目录

1. 支持的部署形态
2. 只读预检
3. Python 与 PostgreSQL 准备
4. Backend 持久配置
5. 前台验收
6. Ubuntu systemd 安装
7. 远程访问与认证
8. 完整验收
9. 更新与回滚
10. 常见故障

## 1. 支持的部署形态

- 开发/验证：仓库内 `.venv`，前台运行 `llmperf-backend`。
- 持久 Ubuntu：复用 `deploy/systemd/llmperf-backend.service.template`，渲染后安装为
  系统 unit，但进程使用普通用户。
- PostgreSQL 是唯一运行数据库。当前没有 checked-in Docker/Kubernetes 部署资产；
  用户只要求部署时不要临时发明另一套架构。

## 2. 只读预检

先解析实际环境，不要假定路径、用户或服务状态：

```bash
pwd
git status --short
python3 --version
psql --version
systemctl --version
ss -ltn
./.venv/bin/llmperf-backend config path
systemctl status --no-pager llmperf-backend.service
```

允许不存在 `.venv`、config 或 unit。记录现有 unit 内容和数据库目标；覆盖已有部署或
修改已有数据库前先确认其所有者、用途、备份和回滚方案。

## 3. Python 与 PostgreSQL 准备

项目要求 Python 3.9+。按目标 Ubuntu 版本确认包名后安装 Python venv、编译依赖、
PostgreSQL server/client；执行 `apt`、创建系统用户或启停 PostgreSQL 前需要相应授权。

仓库环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

按实际 adapter 追加 `.[litellm]`、`.[sagemaker]` 或 `.[correctness]`，不要安装无关 extra。

全新本地数据库的最小路径：

```bash
createdb llmperf
psql -v ON_ERROR_STOP=1 -d llmperf -f sql/postgresql/init.sql
./.venv/bin/llmperf-backend config set DATABASE_URL postgresql+asyncpg:///llmperf
```

生产部署优先使用专用、非 SUPERUSER 数据库角色，并用明确 host/database/role 的
`postgresql+asyncpg://...` URL。不要把密码显示在诊断输出。`create_all` 不升级旧 schema；
更新已有数据库前审查 SQL 差异并备份，项目当前没有 Alembic 自动迁移保证。

## 4. Backend 持久配置

必须以最终 service user 执行配置命令，配置保存在该用户的
`~/.config/llmperf/backend.env`（目录 `0700`、文件 `0600`）：

```bash
./.venv/bin/llmperf-backend config set LLMPERF_SERVER_HOST 127.0.0.1
./.venv/bin/llmperf-backend config set LLMPERF_SERVER_PORT 8000
./.venv/bin/llmperf-backend config list
```

Provider URL/key 也放在 Backend config。key 只用 `--stdin`，unit 不写 `Environment=`。
`llmperfctl config` 是客户端 URL/auth 配置，不能代替 `llmperf-backend config`。

配置优先级是进程环境、`LLMPERF_ENV_FILE`、用户持久配置、工作目录 `.env`。配置改变后
重启 Backend，并用 health/config path 验证实际加载来源。

## 5. 前台验收

安装 systemd 前从仓库根目录启动：

```bash
./.venv/bin/llmperf-backend config list
./.venv/bin/llmperf-backend
```

在另一终端运行 `./.venv/bin/llmperfctl health`。确认成功后正常终止前台进程。此阶段
优先解决 import、配置、PostgreSQL、端口占用和 artifact 权限问题。

## 6. Ubuntu systemd 安装

模板中的三个占位符必须解析为绝对仓库路径、普通用户和组。下面命令假定当前用户就是
已经选定的 service user；使用专用账号时改成经过核对的明确值。不要覆盖变量 `HOME`：

```bash
llmperf_root="$PWD"
llmperf_user="$(id -un)"
llmperf_group="$(id -gn)"
sed \
  -e "s|@LLMPERF_ROOT@|$llmperf_root|g" \
  -e "s|@LLMPERF_USER@|$llmperf_user|g" \
  -e "s|@LLMPERF_GROUP@|$llmperf_group|g" \
  deploy/systemd/llmperf-backend.service.template \
  | sudo tee /etc/systemd/system/llmperf-backend.service >/dev/null
sudo chmod 0644 /etc/systemd/system/llmperf-backend.service
sudo systemd-analyze verify /etc/systemd/system/llmperf-backend.service
sudo systemctl daemon-reload
sudo systemctl enable --now llmperf-backend.service
sudo systemctl status --no-pager llmperf-backend.service
```

存在同名 unit 时先读取并获得覆盖授权，必要时保存可恢复副本。模板的
`WorkingDirectory` 提供项目 `.env` fallback，`ExecStart` 使用现有 `.venv`；
`KillMode=control-group` 保证停止服务时清理 Worker/Ray 子进程。

## 7. 远程访问与认证

默认 `127.0.0.1:8000` 最安全。跨主机访问优先把 Backend 保持在 loopback，由提供
HTTPS 的反向代理转发，并限制防火墙来源。JWT 只签名不加密，不能替代 TLS。

开启固定公钥认证前生成专用 RSA 3072+ 密钥，私钥权限必须 `0600`，Backend 只持有
公钥：

```bash
mkdir -p ~/.config/llmperf/keys
openssl genpkey -algorithm RSA \
  -pkeyopt rsa_keygen_bits:3072 \
  -out ~/.config/llmperf/keys/ctl-private.pem
chmod 0600 ~/.config/llmperf/keys/ctl-private.pem
openssl pkey \
  -in ~/.config/llmperf/keys/ctl-private.pem \
  -pubout \
  -out ~/.config/llmperf/keys/ctl-public.pem
./.venv/bin/llmperf-backend config set LLMPERF_AUTH_ENABLED true
./.venv/bin/llmperf-backend config set LLMPERF_AUTH_KEY \
  /absolute/path/to/ctl-public.pem
```

重启 Backend；issuer/audience 必须与 CLI 一致。远端 CLI 单独配置：

```bash
llmperfctl config set LLMPERF_URL https://llmperf.example.com
llmperfctl config set LLMPERF_PRIVATE_KEY /absolute/path/to/ctl-private.pem
llmperfctl config list
```

不要未经认证直接把 Uvicorn 绑定到公网 `0.0.0.0`。证书和反向代理不由当前 systemd
模板管理，用户需要时必须明确选择并配置该边界。

## 8. 完整验收

```bash
llmperfctl health
llmperfctl scheduler status
llmperfctl planner runtime
llmperfctl provider list
llmperfctl provider models PROVIDER_ID
llmperfctl runner start -f examples/example-smoke.yaml -w
```

记录 smoke Runner ID，并用 `runner status ID --summary` 验证请求数和 outcome。失败时再
调用 `runner logs ID`。`/models` 成功不代表 inference 成功；smoke 可能产生 Provider
费用，只有在部署验收范围包含推理调用时执行。

## 9. 更新与回滚

更新前检查 dirty worktree、当前 commit、数据库备份、unit 和 config 路径。不要覆盖用户
改动。代码/依赖更新后重新运行 `.venv/bin/python -m pip install -e .`，审查 schema
变化，再重启：

```bash
sudo systemctl restart llmperf-backend.service
sudo systemctl status --no-pager llmperf-backend.service
sudo journalctl -u llmperf-backend.service -n 200 --no-pager
```

若仓库位置或 service user 改变，重新渲染 unit 并 `daemon-reload`。回滚必须同时考虑
代码、依赖和数据库 schema；没有验证过向后兼容时不要只回退代码。

## 10. 常见故障

- unit 启动失败：检查 `User/Group`、绝对 `WorkingDirectory/ExecStart`、`.venv` 和权限。
- 数据库失败：检查 URL driver 必须是 `postgresql+asyncpg`、socket/host、角色和 schema。
- 端口失败：检查 `ss -ltn` 与 `LLMPERF_SERVER_HOST/PORT`。
- 配置未生效：确认命令由 service user 执行、config path 正确且服务已重启。
- Worker import 失败：确认 editable install 指向当前仓库并重新安装项目。
- Provider 失败：先查 public Profile 和 catalog，再查 smoke summary/logs 的首个 HTTP 错误。
- 停服后残留 Worker：确认实际 unit 保留 `KillMode=control-group`。
