# PhysAgent

基于物理约束的多智能体材料发现系统

PhysAgent 是一个闭环自动化材料发现系统，通过 LangGraph 编排多个 AI Agent 协作，结合机器学习势函数（MACE-MP-0/CHGNet）进行真实的材料性质计算，实现从假设生成到性质验证的全自动迭代优化。系统在闭环结束后自动生成以发现材料为主题的学术论文草稿。

## 研究背景

当前 AI for Materials 的工作大量集中在单点模型（性质预测、结构生成），但缺少一个能自主完成"提出假设 → 计算验证 → 分析结果 → 迭代改进"全流程的系统。具体而言，现有系统存在三个空白：

1. **缺闭环**：MatPilot、AtomAgents 等系统演示了单次发现，但没有自动迭代修正能力
2. **缺物理约束**：现有 Agent 多依赖 LLM 文本自反思进行纠错，而非基于数值物理量的反馈驱动
3. **缺深度计算**：现有 Agent 多为 API 调用，不涉及真实的结构弛豫和性质计算

PhysAgent 针对这些空白，构建了首个物理约束驱动的多智能体材料发现闭环系统。其核心创新在于：用数值物理量（生成能、带隙、力收敛等）而非文本反思驱动 Agent 自修正，结合 ML 势函数（MACE-MP-0）实现真实的材料性质计算，使系统能够自主迭代逼近目标材料。

## 系统架构

PhysAgent 采用 LangGraph 状态机编排 4 个 Agent 节点，形成闭环迭代：

```
Planner Agent ──→ Simulator Agent ──→ Evaluator Agent ──→ Controller
  (DeepSeek-R1)     (MACE-MP-0)        (物理约束验证)      (数值路由)
  RAG文献检索        Pymatgen构建        硬约束: 电荷中性     PASS → 输出候选
  假设生成           ASE结构弛豫          原子间距/力收敛      REVISE → 回Planner
  JSON输出           性质计算             带隙/生成能检查      REJECT → 换方案
       ↑                                                        │
       └────────────────── 物理反馈 ────────────────────────────┘
```

### Agent 详细说明

**Planner Agent（规划者）**
- 使用 DeepSeek-R1 推理模型，temperature=0.7，兼顾创造性与合理性
- 通过 RAG 检索 ChromaDB 中的学术文献，基于文献证据生成候选材料假设
- 输出结构化 JSON：化学式（composition）、空间群（space_group）、推理过程（reasoning）
- 在后续迭代中接收 Evaluator 的反馈和历史记录，避免重复探索失败组分

**Simulator Agent（模拟者）**
- 使用 Pymatgen 构建钙钛矿 ABX3 晶体结构（支持纯组分和混合组分超胞）
- 通过 ASE + MACE-MP-0 机器学习势函数进行结构弛豫（BFGS 优化器，fmax=0.05 eV/Å）
- 计算材料性质：总能量、生成能（基于元素参考态）、力、应力
- 使用已知端点插值法估算带隙（bandgap proxy）

**Evaluator Agent（评估者）**
- 执行物理硬约束检查（任一不通过则 REJECT）：
  - 电荷中性：氧化态总和 ≈ 0
  - 原子间距：最小距离 > 1.0 Å
  - 生成能范围：-5.0 ~ +2.0 eV/atom
  - 力收敛：最大力 < 0.05 eV/Å
- 软约束：查询 Materials Project 的 energy above hull（仅供参考，不触发 REJECT）
- 目标属性检查：对比计算值与用户设定的目标范围（bandgap proxy 有 ±0.3 eV 容差）
- 使用 LLM 生成改进建议文本，反馈给 Planner

**Controller（自修正控制器）**
- 纯 Python 逻辑，不使用 LLM
- 基于数值物理量（而非文本反思）进行路由决策：
  - `PASS` → 结束，输出候选材料
  - `REVISE` → 返回 Planner 调整组分
  - `REJECT` → 返回 Planner 换全新方案
  - 模拟错误 → 重试一次 Simulator，再失败则返回 Planner
- 追踪 composite score（目标距离之和），连续 3 轮无改善则 early stop
- 所有属性缺失时 score 为 inf，防止假阳性 PASS

**Writer Agent（论文生成）**
- 闭环结束后自动运行，基于发现轨迹 + RAG 文献生成完整学术论文
- 论文以发现的材料为主题（非系统本身），包含摘要、引言、计算方法、结果讨论、结论、参考文献
- 支持长论文分段生成（自动检测截断并续写）

### 知识增强（RAG）

系统支持两种 RAG 模式：

- **本地模式**（默认）：使用预下载的论文库（`rag_repo/`），通过 ChromaDB 向量检索
- **云端增强模式**（`--enrich`）：运行前根据 goal 自动搜索 arXiv/PubMed 相关论文，下载并增量更新知识库

RAG 知识库使用 `BAAI/bge-m3` 嵌入模型，chunk size=1000，top-k=6 检索。

## 快速开始

### 环境要求

- Python 3.12+
- CUDA（可选，用于加速 ML 势函数计算）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：LangGraph、LangChain、Pymatgen、ASE、MACE-torch、CHGNet、ChromaDB、mp-api

### 2. 配置 API Key

创建 `.env` 文件（参考 `.env.example`）：

```
DEEPSEEK_API_KEY=sk-your-key-here
MP_API_KEY=your-mp-api-key-here
```

| Key | 必需 | 用途 |
|-----|------|------|
| `DEEPSEEK_API_KEY` | 是 | Planner (R1) 和 Evaluator/Writer (chat) 的 LLM 调用 |
| `MP_API_KEY` | 否 | 查询 Materials Project 数据库（energy above hull、已知带隙等）。可在 https://next-gen.materialsproject.org/ 免费注册获取。缺失时系统使用内置的元素参考能量表作为 fallback |

### 3. 构建 RAG 知识库

```bash
# 下载预定义论文库到 rag_repo/（5 个研究领域，约 50 篇论文）
python download.py
```

首次运行 PhysAgent 时会自动将论文库索引到 ChromaDB 向量数据库（`chroma_db/`）。

### 4. 运行

```bash
# 默认目标：寻找带隙 1.3-1.6 eV、生成能 < 0 的钙钛矿
python -m physagent.main

# 启用云端论文搜索增强 RAG 知识库
python -m physagent.main --enrich

# 自定义目标
python -m physagent.main \
    --goal "Find lead-free ABX3 perovskite with bandgap 1.3-1.6 eV" \
    --max-iterations 10
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--goal` | 寻找带隙 1.3-1.6 eV 的钙钛矿 | 材料发现目标（自然语言，英文效果更好） |
| `--target` | bandgap_proxy [1.3, 1.6] + formation_energy < 0 | 目标属性 JSON 字符串 |
| `--max-iterations` | 20 | 最大迭代次数 |
| `--enrich` | 关闭 | 启用云端论文搜索增强 |

### 示例场景

```bash
# 无铅钙钛矿（光伏）
python -m physagent.main \
    --goal "Find a lead-free ABX3 perovskite with bandgap 1.3-1.6 eV using only Sn or Ge at B site"

# 宽带隙（叠层电池顶电池）
python -m physagent.main \
    --goal "Find ABX3 with wide bandgap 1.7-1.9 eV for tandem solar cell" \
    --target '{\"bandgap_proxy\": {\"min\": 1.7, \"max\": 1.9}, \"formation_energy\": {\"max\": 0}}'

# 蓝光 LED
python -m physagent.main \
    --goal "Find ABX3 perovskite with bandgap 2.5-3.0 eV for blue LED" \
    --target '{\"bandgap_proxy\": {\"min\": 2.5, \"max\": 3.0}, \"formation_energy\": {\"max\": 0}}' \
    --enrich

# 高稳定性优先
python -m physagent.main \
    --goal "Find the most stable ABX3 halide perovskite with bandgap in visible range" \
    --target '{\"bandgap_proxy\": {\"min\": 1.5, \"max\": 2.5}, \"formation_energy\": {\"max\": -0.5}}'
```

### 输出

每次运行在 `results/` 目录生成：
- `run_<timestamp>.json` — 完整迭代轨迹数据（每轮的组分、计算属性、评估结果、分数）
- `paper_<timestamp>.md` — 学术论文草稿（摘要、引言、计算方法、结果讨论、结论、参考文献）

## 支持的材料体系

当前支持卤化物钙钛矿 ABX3：

| 位点 | 可选元素 | 说明 |
|------|---------|------|
| A 位 | Cs, Rb, K | 无机阳离子 |
| B 位 | Pb, Sn, Ge, Ti, Zr | 二价金属阳离子 |
| X 位 | I, Br, Cl, F | 卤素阴离子 |

支持混合组分（如 CsSn0.5Pb0.5I3、CsPbBr1.5Cl1.5），通过 2×1×1 超胞有序替换实现。化学式格式要求：简单格式（如 `CsPbI3`、`CsSn0.5Ge0.5I3`），不支持括号格式。

### 带隙估算方法

ML 势函数（MACE/CHGNet）不直接预测带隙。系统使用基于已知实验值的插值法：

| 组分 | 实验带隙 (eV) |
|------|--------------|
| CsPbI3 | 1.73 |
| CsPbBr3 | 2.30 |
| CsPbCl3 | 2.90 |
| CsSnI3 | 1.30 |
| CsSnBr3 | 1.75 |
| CsGeI3 | 1.60 |
| ... | ... |

对于混合组分，按各端点的占比加权插值。精度约 ±0.3 eV，Evaluator 在目标检查时自动应用此容差。

## 项目结构

```
physagent/
├── config.py                 # 配置 + AgentState TypedDict + compute_score
├── main.py                   # LangGraph 编排 + CLI 入口 + enrich_knowledge_base
├── agents/
│   ├── planner.py            # 假设生成（DeepSeek-R1 + RAG 检索）
│   ├── simulator.py          # 结构构建 → ML势弛豫 → 性质计算
│   ├── evaluator.py          # 物理硬约束 + 软约束 + 目标检查
│   ├── controller.py         # 数值路由 + 历史追踪 + Early Stop
│   └── writer.py             # 学术论文自动生成
├── tools/
│   ├── structure_builder.py  # Pymatgen 钙钛矿结构构建（纯组分 + 超胞混合）
│   ├── ml_potential.py       # MACE/CHGNet 弛豫 + 性质计算 + 带隙插值
│   └── database_query.py     # Materials Project API 查询 + 元素参考能量 fallback
├── memory/
│   └── rag_store.py          # ChromaDB 向量库构建/加载/增量更新 + RAG 链工厂
└── paper_fetcher.py          # 多源论文搜索（arXiv + Semantic Scholar + PubMed）

tests/                        # 58 个测试（单元测试 + 集成测试）
download.py                   # 独立的批量论文下载器
```

## 测试

```bash
# 运行全部 58 个测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_controller.py -v

# 运行单个测试函数
python -m pytest tests/test_evaluator.py::test_verdict_pass -v
```

测试覆盖：
- `test_state.py` — compute_score 边界情况（7 tests）
- `test_controller.py` — 路由逻辑、历史追踪、retry 机制（12 tests）
- `test_evaluator.py` — 物理约束检查、目标检查、verdict 判定（14 tests）
- `test_structure_builder.py` — 组分解析、结构构建、混合组分（10 tests）
- `test_ml_potential.py` — 带隙插值估算（4 tests）
- `test_database_query.py` — 元素参考能量、生成能计算（4 tests）
- `test_planner.py` — JSON 解析、历史格式化（5 tests）
- `test_integration.py` — 完整 LangGraph 闭环（mock Agent，2 tests）

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 编排 | LangGraph | 状态机图定义、条件路由、闭环迭代 |
| LLM | DeepSeek-R1 | Planner 假设生成（推理模型） |
| LLM | DeepSeek-chat | Evaluator 建议生成、Writer 论文撰写 |
| ML 势函数 | MACE-MP-0 | 结构弛豫、能量/力/应力计算（主选） |
| ML 势函数 | CHGNet | 备选计算引擎（支持磁矩） |
| 结构操作 | Pymatgen | 晶体结构构建、组分分析、氧化态推断 |
| 原子模拟 | ASE | BFGS 优化器、计算器接口 |
| 向量检索 | ChromaDB + bge-m3 | 论文知识库存储与语义检索 |
| 材料数据 | Materials Project API | 已知材料性质查询、凸包距离参考 |
| 论文搜索 | arXiv + PubMed | 云端论文搜索与下载（--enrich 模式） |
