# 双卡 SGLang 模型服务

本仓库用于在分配了两张 PPU-ZW810E 卡的 Linux 容器中，通过 SGLang 以张量并行（TP=2）运行 `Qwen/Qwen3.8-27B`，并用合成请求维持推理负载。目标利用率默认是两卡平均约 60%，控制器每 10 秒按两卡平均利用率调整并发数；这不是对单卡利用率的保证。

## 给另一容器中的 AI 的部署指引

请先读完本节，再在目标容器内检查环境。不要因为仓库里有启动脚本，就假定模型、依赖、显卡或磁盘已经就绪。

### 1. 检查容器环境

在仓库目录执行：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
/usr/local/bin/python --version
/usr/local/bin/python -m pip show sglang torch
test -x /usr/local/bin/sglang && echo 'SGLang CLI found'
df -h /root
```

运行条件：容器内可见两张可用的 PPU；系统 Python 中已有可运行的 SGLang 和 PyTorch；`/root` 有至少约 60 GB 可用空间；容器能访问 ModelScope。参考环境为系统 Python 3.12.3、SGLang 0.5.12、Torch 2.10。**推理服务使用系统安装的 SGLang/Torch，不要切换到 Conda 的 `sglang` 环境，也不要无故升级系统依赖。**

如果从 GitHub 取得本仓库：

```bash
git clone https://github.com/wangkang1019/run_sglang_ly.git
cd run_sglang_ly
chmod +x start_sglang_2gpu.sh stop_sglang_2gpu.sh
```

如果这三个脚本已在当前工作目录，就直接在该目录继续。

### 2. 确认 ModelScope 下载工具

启动脚本在模型缺失时调用 ModelScope 的 `ms-hub`，默认路径是 `/root/miniconda3/envs/sglang/bin/ms-hub`。它只负责下载模型，SGLang 推理仍使用系统 Python。

```bash
test -x /root/miniconda3/envs/sglang/bin/ms-hub && \
  /root/miniconda3/envs/sglang/bin/ms-hub --version
```

如果工具不存在、但上述 Conda Python 存在，可从官方 PyPI 安装，不要使用可能需要内部凭据的默认私有 PyPI 源：

```bash
/root/miniconda3/envs/sglang/bin/python -m pip install \
  --index-url https://pypi.org/simple modelscope-hub
```

如果 `ms-hub` 安装在其他位置，启动时用 `MODELSCOPE_CLI=/实际路径/ms-hub` 指定。

### 3. 启动

默认模型为 BF16 的 `Qwen/Qwen3.8-27B`，模型文件约 55.6 GB。默认保存到容器本地的 `/root/models/Qwen3.8-27B`。本地盘不是持久化盘；容器或 Pod 重建后模型可能消失，之后启动会重新下载。不要把模型下载到空间不足的持久化工作区。

```bash
./start_sglang_2gpu.sh /root/models/Qwen3.8-27B
```

启动命令会很快返回，后台 supervisor 会先检查配置、分词器、权重索引和所有索引权重分片：

- 模型已完整：跳过下载，直接启动 SGLang。
- 模型不存在或不完整：从 ModelScope 下载 `Qwen/Qwen3.8-27B`，默认 4 路并发；文件完整校验通过后才启动 SGLang。
- 下载未完成或失败：不会启动服务。查看日志后重新运行同一条命令，会复用已校验文件并重试缺失文件。

下载进度、请求负载控制日志和服务日志：

```bash
tail -f logs/runner.log logs/server.log
```

在 `runner.log` 中等待 `Model download completed`（若需要下载）和 `SGLang is ready`。确认服务健康：

```bash
curl --noproxy '*' -sS -o /dev/null -w 'HTTP %{http_code}\n' \
  http://127.0.0.1:30001/health
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

健康检查返回 `HTTP 200` 表示服务已就绪。启动后脚本会自动发送本地合成请求，回复会被读取后丢弃，不保存为对话结果；日志会记录利用率、请求计数和服务状态。

### 4. 停止、重启与常用设置

```bash
./stop_sglang_2gpu.sh
./start_sglang_2gpu.sh /root/models/Qwen3.8-27B
```

停止命令会停止当前下载器或 SGLang 服务。断开 SSH 不会停止后台进程；容器/Pod 被停止或回收时，服务会结束。

常用环境变量：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `SGLANG_HTTP_PORT` | `30001` | HTTP 服务端口 |
| `MODELSCOPE_MODEL_ID` | `Qwen/Qwen3.8-27B` | 模型仓库 ID，格式为 `owner/name` |
| `MODELSCOPE_CLI` | `/root/miniconda3/envs/sglang/bin/ms-hub` | ModelScope 下载程序路径 |
| `MODELSCOPE_MAX_WORKERS` | `4` | ModelScope 文件下载并发数 |
| `TARGET_GPU_UTIL` | `60` | 目标两卡平均利用率（百分比） |
| `UTIL_HYSTERESIS` | `10` | 目标上下的调整区间（百分点） |
| `INITIAL_CONCURRENCY` | `2` | 初始推理请求并发数 |
| `MAX_CONCURRENCY` | `16` | 最大推理请求并发数 |
| `MAX_TOKENS` | `256` | 合成请求最大生成 token 数 |
| `SGLANG_CONTEXT_LENGTH` | `8192` | 服务上下文长度 |

若需调整 HTTP 端口：

```bash
SGLANG_HTTP_PORT=30002 ./start_sglang_2gpu.sh /root/models/Qwen3.8-27B
```

改端口后，健康检查 URL 也要改为相同端口。**不要把 HTTP 端口设置到 `SGLANG_PORT`**；SGLang 会用它分配内部通信端口。若出现端口占用，先查 `ss -ltnp | grep 30001`，再选择空闲的 `SGLANG_HTTP_PORT`。

### 5. 网络与故障排查

- 服务监听 `0.0.0.0:30001`，默认未配置 API 鉴权。只按容器平台的网络策略开放端口，不要直接暴露到不可信公网。
- 下载或启动异常先看 `logs/runner.log` 与 `logs/server.log`；下载未完成时重新执行启动命令以续传/重试。
- 若报 `address already in use`，检查当前 HTTP 端口占用并改 `SGLANG_HTTP_PORT`，不要杀不明进程。
- 若报模型文件缺失，不要手动删除下载目录；检查网络和剩余空间后重新执行同一启动命令。
- 若 `nvidia-smi` 看不到两张卡，先由容器平台确认两卡确实已分配给本容器，再启动服务。

`./start_sglang_2gpu.sh --help` 可查看全部运行参数；`常用命令.txt` 收录了日常操作命令。
