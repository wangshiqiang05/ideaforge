# 🛠️ IdeaForge

> **输入一个 idea，自动帮你做出可运行的 Python 项目。**

---

## 1. 🎯 是什么

IdeaForge 是一个自动化代码生成工具。你只需输入一句想法（比如"一个带 SQLite 存储的 CLI 待办应用"），它会：

1. 🔍 去 GitHub 找相似项目并克隆下来
2. 🧩 把多个仓库的代码融合成一个新方案
3. 🔁 让多个大模型轮流**审查 → 修复**，反复打磨代码（类似 GAN 的对抗循环）
4. 📊 最后给你一份验证报告

**当前版本（v1）只支持 Python 项目。**

---

## 2. 💡 为什么用它

| 你平时怎么做 | 用 IdeaForge |
|---|---|
| 自己搜 GitHub、读代码、拼凑 | 🔍 自动检索 + 克隆 + 融合 |
| 写完代码自己 review 容易漏 | 👥 三个审查者 + 终审交替检查 |
| 改完不知道有没有问题 | 🧪 容器内自动跑语法检查、ruff、pytest |
| 花多少 token 心里没底 | 💰 预算熔断，超了就暂停问你 |

**安全**：所有第三方代码都在 Docker 沙箱里跑（禁网络、限内存 4G、限 2 核 CPU），不会污染你的机器。

---

## 3. 🚀 怎么做

### 3.1 📋 准备

只需要三样东西：

- 🐍 **Python ≥ 3.11**
- 🐳 **Docker**（必需，所有代码执行都在容器里）
- 🐙 **Git**

### 3.2 📦 安装

```bash
cd ideaforge
pip install -e ".[dev]"
```

### 3.3 🔑 设置 API key

key **只能**放在环境变量里，不能写进配置文件。

```bash
# 按你用的模型设置
export DEEPSEEK_API_KEY="<your-key>"
export OPENAI_API_KEY="<your-key>"
export GITHUB_TOKEN="<your-token>"   # 可选，能提升 GitHub 搜索限额
```

Windows PowerShell 写法：`$env:DEEPSEEK_API_KEY = "<your-key>"`

### 3.4 ⚙️ 生成配置

```bash
cp config.example.yaml config.yaml
```

然后编辑 `config.yaml`，最少要填两处：

- **`models`**：填你想用的模型（每个含 `model`、`base_url`、`api_key_env`）
- **`roles`**：把 8 个角色分配给模型

> 💡 8 个角色是：`fixer`、`reviewer_runnable`、`reviewer_style`、`reviewer_idea`、`judge_final`、`prospector`、`architect`、`synthesizer`
>
> ⚠️ 常见坑：如果配置校验不过，`POST /tasks` 会返回 400 `invalid_config`，`items` 里会逐条列出哪里错了，照着改即可。

### 3.5 🏁 启动

```bash
python -c "import uvicorn; from ideaforge.api.main import create_app; uvicorn.run(create_app(config_path='config.yaml'), host='127.0.0.1', port=8000)"
```

启动后打开 <http://127.0.0.1:8000/docs> 就能看到所有接口。

> ⚠️ 不要用 `uvicorn ideaforge.api.main:create_app --factory`，那样**不会**加载 `config.yaml`，会因模型池为空而创建任务失败。

### 3.6 📝 提交一个任务

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"idea": "a CLI todo app with SQLite storage"}'
```

成功会返回：

```json
{"task_id": "f-xxxxxxxx", "status": "running"}
```

失败会返回 400，`items` 里是逐条错误，照着改配置再来。

### 3.7 👀 看进度和结果

```bash
# 查状态
curl http://127.0.0.1:8000/tasks/f-xxxxxxxx

# 实时看事件流（类似 tail -f）
curl -N http://127.0.0.1:8000/tasks/f-xxxxxxxx/events

# 任务结束后拿报告
curl http://127.0.0.1:8000/tasks/f-xxxxxxxx/report
```

### 3.8 ✋ 遇到审批 / 预算熔断怎么办

这两类情况任务会**暂停等你拍板**：

**① 融合方案待审批**（`approval_required`）

```bash
curl -X POST http://127.0.0.1:8000/tasks/f-xxxxxxxx/approval \
  -H "Content-Type: application/json" \
  -d '{"decision": "approve", "gpl_accepted": false}'
```

- `decision`：`approve` 或 `reject`（reject = 终止任务）
- 涉及 GPL 项目时必须显式 `gpl_accepted: true`

**② 预算用完**（`budget_break`）

```bash
curl -X POST http://127.0.0.1:8000/tasks/f-xxxxxxxx/budget-decision \
  -H "Content-Type: application/json" \
  -d '{"decision": "raise_budget", "new_budget": 10.0}'
```

- `raise_budget`：加钱继续（带上新预算）
- `abort`：终止任务，已有的产出还是可以从 report 端点拿

### 3.9 📊 任务状态一览

| 状态 | 意思 |
|---|---|
| 🟢 `running` | 正在跑 |
| ✋ `awaiting_approval` | 等你审批 |
| 💰 `awaiting_budget_decision` | 等预算决策 |
| ✅ `completed` | 完成 |
| 🟡 `degraded` | 轮次用完还没收敛，**诚实降级**（报告会标注未通过完整验证） |
| ❌ `failed` | 失败 |

---

## 4. 🧪 验证是否正常

```bash
cd ideaforge && python -m pytest tests -q
python -m ruff check .
```

---

## 5. ⚠️ v1 的已知限制

1. 🐍 **只支持 Python 项目**（多语言是 v2 的计划）
2. 🔒 **只读 GitHub**：只搜和克隆，不 push、不开 PR
3. 🖥️ **UI 暂不可一键打开**：`ui/index.html` 没挂到服务上，请直接用上面第 3 章的接口
4. 🗄️ **没有"仓库使用经验"缓存**（原规格有，当前缺失，待裁决）
