# 项目说明：contract-qa-agent（合同条款智能问答）

一个**引用可核对**的合同条款 RAG 系统。核心目标不是「能答」，而是「答得能被逐字核对」：
每条回答必须给出出处（哪份文件、哪一章、哪一条、第几页）。

## 技术栈

- Python 3.11+，包管理用 **uv**（勿用 pip 手动装包）
- LangGraph 状态机、Chroma 向量库、PyMuPDF、rank-bm25、Streamlit、structlog、pydantic v2
- LLM：DeepSeek（`deepseek-chat`）；Embedding：硅基流动 `BAAI/bge-m3`（1024 维）
- 构建：hatchling；布局：src layout（包在 `src/contract_qa`）

## 常用命令（Windows PowerShell 环境）

```powershell
uv sync
uv run contract-qa check
uv run contract-qa parse
uv run contract-qa index
uv run contract-qa ask "..."
uv run pytest
uv run ruff check .
```

纯标准库回退路径（PowerShell 下）：

```powershell
$env:PYTHONPATH = "src"; python -m contract_qa check
```

## 目录结构

```text
src/contract_qa/
  text.py           PDF 取文、页眉页脚与目录清洗、标题分级
  splitter.py       条款切分（article / outline / chunk 三种模式）
  parse.py          解析编排与 JSONL 输出
  index.py          BM25 与 Chroma 向量索引
  retrieval.py      双路召回 + 加权 RRF 融合
  graph.py          五节点状态机（route/rewrite/retrieve/generate/verify）
  llm.py            OpenAI 兼容客户端（向量化 + 对话）
  config.py         .env 加载与配置聚合
  cli.py            命令行入口
scripts/            数据获取、验收与冒烟测试
tests/              自动化测试
docs/               范围、设计与验收记录
data/raw/           原始 PDF（不入库）
data/processed/     条款、索引与产物（不入库）
```

## 硬性约束

1. `.env` 含密钥，已被 `.gitignore` 排除，不得提交，不得在代码/输出中展示密钥内容。
2. `data/raw/` 与 `data/processed/` 的产物不入版本库。
3. **指标纪律**：Recall@5、引用正确率等指标只记录实测结果；计划书中的数字仅是示例，
   不得当作实测值写入 README、简历或文档。
4. 命名坦白：当前实现是固定五节点链路（RAG pipeline），不是自主 Agent；
   verify 回退已具备，但工具调用与自主决策还没有，所以仍不称 Agent。
   等 D6 加上 verify 回退与工具调用后才可改称 Agent。
5. `BM25_WEIGHT=1.0 / DENSE_WEIGHT=1.0`（1:1）是 D5 三轮评测的实测结论：
   权重比超过约 1.31（RRF k=60、召回深度 20）会压制单路命中的答案；调权重需重跑
   `scripts/evaluate_retrieval.py` 留数据；非法权重值必须直接报错而不是静默生效。

## 当前状态

D1–D6 已完成（骨架、解析切分、索引检索、四节点链路、检索评测、引用校验）。
D6 verify 节点：规则校验 + 模型核对，不通过最多回退 2 轮，记录见 `docs/d6-smoke.md`。
下一步待定：结构化抽取（表格重建、子合同分段）或工具调用。详细进度与数据见 `docs/` 下各文档。
