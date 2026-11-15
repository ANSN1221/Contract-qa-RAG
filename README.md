# 合同条款智能问答

一个**引用可核对**的 RAG 系统。核心不是「能答」，而是「答得能被逐字核对」：
每条回答都必须给出它出自哪份文件、哪一章、哪一条、第几页。

和常见的 RAG demo 差别在三点：切分以条款为最小单位（不是按字数切）、
检索用 BM25 + 向量双路加权融合（不是只用向量）、引用带页码可翻页验证（不是只报文件名）。

> 关于命名：仓库已更名为 `Contract-qa-RAG`，与实现一致——当前是**固定的五节点链路**，
> 模型没有工具调用与自主决策。若后续加上工具调用，才适合再叫 Agent。

## 当前状态

**D4：主链路改为状态机。** route（判断问题是否属于合同域）→ rewrite（改写查询、拆复合问题）
→ retrieve（双路召回 + RRF）→ generate（严格 JSON，malformed 自动重试一次）。
节点级降级：任一接口调用失败都会按预设路径继续（放行 / 退回原句 / 输出降级答案），
引用不一致（候选外引用、可答但无引用等）会标记「需要人工复核」而非静默输出。
冒烟结果见 `docs/smoke.md`，解析验收见 `docs/parse-acceptance.md`。

**D5：检索评测与重排。** 50 条种子题（fact 17 / cross 17 / unanswerable 16）测 Recall@5：
仅 BM25 79%、仅向量 85%、混合 RRF 88%、混合 + rerank 88%（默认权重 1:1）。
切分规整：569 → 766 条，>500 字条款清零、极短条目归并，页码区域命中 94.0% 持平。
三个发现：① 默认权重比 2:1 超过 RRF 的压制阈值 1.31，会把单路命中的答案挤到 20 名外；
② rerank 四轮零召回增益，只能重排池内条款；③ 四道边界题（同构表格、表格碎片、标题正文被切断），
需字段级结构化抽取才能解决。数据见 `docs/eval-d5.md`，分析见 `docs/d5-findings.md`。重排默认关闭。

**D6：引用校验与回退。** 生成后加 verify 节点：先跑零成本规则校验（可答却没引用、
引用不在候选里、答案为空），再让模型核对答案是否被引用条款支持。
不通过带上问题说明回退重生成，最多 2 轮；校验接口本身失败时放行并记入 node_errors。
冒烟 3 题：2 题可答且引用正确、1 题正确拒答，均一次通过，可答题均耗时约 5.3 秒
（verify 的模型校验约 1 秒，拒答题不触发模型校验）。
记录见 `docs/d6-smoke.md`。

进度：D1 骨架 ✓ · D2 解析与切分 ✓ · D3 索引与检索 ✓ · D4 四节点主链路 ✓
· D5 检索评测 ✓ · D6 引用校验 ✓

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/why-custom-splitter.md](docs/why-custom-splitter.md) | 为什么不用现成切分器：对照实验、真实样例、代价与适用边界 |
| [docs/problems/](docs/problems/) | D1–D3 的 16 个真实卡点：现象、原因、处理、教训（PDF） |
| [docs/parse-acceptance.md](docs/parse-acceptance.md) | 解析与切分的抽样验收（766 条条款、页码命中 94.0%） |
| [docs/splitter-comparison.md](docs/splitter-comparison.md) | 现成切分工具与本项目切分器的对照数据 |
| [docs/smoke.md](docs/smoke.md) | D4 冒烟记录（6 题、路由、改写查询、耗时、token） |
| [docs/d6-smoke.md](docs/d6-smoke.md) | D6 冒烟记录（verify 轮次、引用、耗时拆解、未覆盖路径） |
| [docs/d6-verify-overview.pdf](docs/d6-verify-overview.pdf) | D6 说明文档（PDF）：链路位置、三类错误、两层校验、回退与代价 |
| [docs/eval-d5.md](docs/eval-d5.md) | D5 检索评测：四组对照、逐题名次、权重扫描、耗时 |
| [docs/d5-findings.md](docs/d5-findings.md) | D5 结论：RRF 压制阈值、rerank 边界、样本量限制 |
| [docs/mvp-scope.md](docs/mvp-scope.md) | 范围边界：做什么、不做什么、为什么 |

## MVP

- PDF 入库与条款级切分：已完成
- BM25 + 向量混合检索、RRF 融合：已完成
- 单文档问答与强制条款引用：已完成
- 路由、查询改写、复合问题拆解：已完成
- 重排（rerank）与引用校验、最多 2 轮回退：进行中
- 60 条离线评测集与指标报告
- Streamlit 溯源界面

明确不做：扫描件 OCR 全流程、法律意见生成、多人权限、移动端、自建向量数据库集群。
边界理由见 `docs/mvp-scope.md`。

## 快速开始

要求 Python 3.11+ 与 uv。

```bash
uv sync
cp .env.example .env      # 填入 LLM 与 Embedding 密钥
uv run contract-qa check  # 环境与数据检查
uv run contract-qa parse  # PDF -> 条款（输出 data/processed/*.jsonl）
uv run contract-qa index  # 建 BM25 与向量索引
uv run contract-qa ask "宁国市垃圾中转站项目的合同总价是多少？"
uv run pytest
uv run ruff check .
```

未安装 uv 时，纯标准库路径同样可跑通（除向量检索外）：

```bash
PYTHONPATH=src python -m contract_qa check
PYTHONPATH=src python -m contract_qa parse
python -m unittest discover -s tests -v
```

`ask` 支持 `--top-k`（送入模型的条款数，默认 8）与 `--show-hits`（打印双路召回排名与融合分）。

## 模型配置

DeepSeek 只提供对话模型，没有 Embedding 接口，因此需要两家服务：

| 用途 | 服务 | 默认模型 |
| --- | --- | --- |
| 问答生成 | DeepSeek | `deepseek-chat` |
| 向量化 | 硅基流动（OpenAI 兼容） | `BAAI/bge-m3`（1024 维） |

密钥写入 `.env`，该文件已被 `.gitignore` 排除，不会进入版本库。

## 数据

15 份公开政府采购文件（10 份合同 + 5 份招标文件，共 635 页），来源为中国政府采购网、
北京市政府采购网、河南省政府采购网及各地公共资源交易中心。元数据见 `data/document-register.csv`，
下载脚本 `scripts/fetch_public_documents.py`。

其中 5 份为纯扫描件（无可抽取文本），刻意保留为解析失败样本，不参与切分。
PDF 原文不进版本库。

## 目录

```text
src/contract_qa/
  text.py           PDF 取文、页眉页脚与目录清洗、标题分级
  splitter.py       条款切分（article / outline / chunk 三种模式）
  parse.py          解析编排与 JSONL 输出
  index.py          BM25 与 Chroma 向量索引
  retrieval.py      双路召回 + 加权 RRF 融合
  graph.py          五节点状态机（route/rewrite/retrieve/generate/verify）
  llm.py            OpenAI 兼容客户端（向量化 + 对话）
  config.py         .env 加载与配置聚合（密钥、路径、检索权重）
  cli.py            命令行入口
scripts/            数据获取、验收与冒烟测试
tests/              自动化测试
docs/               范围、设计与验收记录
data/raw/           原始 PDF（不入库）
data/processed/     条款、索引与产物（不入库）
```

## 切分规则

条款单元按文档自身的编号体系选择，不强行统一：

- **article**：`第X条` 出现 8 次以上，按条切，章/节写入 `chapter_path`
- **outline**：靠 `一、` / `（一）` / `1、` / `1.1` 分层，从最粗的编号层往下试切，
  取第一条「中位长度落在 60–700 字符」的层级
- **chunk**：没有稳定编号，按 700 字符窗口切，强制带页码

三条硬约束：目录页整页丢弃（否则会留下「第一章」三个字的残行污染章节路径）；
只有标题没有正文的条目并入下一条；超过 500 字符的条款按标点切开，片段继承
父条款编号并写入 `parent_clause_id`；带编号的单行极短条目（<40 字）并入相邻条款。

## 检索策略

双路召回后用加权 RRF 融合，权重可用 `BM25_WEIGHT` / `DENSE_WEIGHT` 调
（写入 `.env` 或进程环境变量均可，默认 1.0 / 1.0；非法值会直接报错而不是静默生效）。

默认 1:1 是 D5 三轮评测的实测结论：权重比超过约 1.31 时，权重高的一路的长尾
会压过另一路的头部，把单路命中的答案挤出候选池（Q06 实测：向量第 1 名被
BM25 第 20 名压制到融合第 21 名）。向量负责语义泛化，关键词负责术语命中。

## 指标纪律

Recall@5、引用正确率、不可答硬答率、P95 延迟与成本只记录实测结果。
计划书中的数字仅是示例，不得直接写入简历或 README。

## 下一步

D6 已完成（verify 节点）。下一步二选一：
- **结构化抽取**：PDF 表格重建（`find_tables`）+ 子合同分段，修 D5 的四道边界题
- **工具调用**：把检索/比对/翻页做成可调工具，才够得上「Agent」这个词

D5 留下的待办：
- 四道边界题（Q20/Q24/Q34/Q43）已证实字段级切分无效（两次实验均低 6 个百分点），需更靠前的结构化处理
- rerank 待召回进一步修好后再复评

当前基线：766 条条款、检索 Recall@5 88%（混合 1:1）、D6 冒烟 3 题（2 可答 / 1 正确拒答）。
可答题端到端约 5.3 秒，其中 verify 的模型校验约 1 秒。

链路从 D3 的单次检索变成四节点后，平均耗时从 1.84 秒升到 3.65 秒——多出来的两次 LLM 调用
（route 与 rewrite）换来的是：无关问题 1.15 秒直接短路不检索、复合问题能拆成两问分别召回。
D6 再加一次 verify 调用，可答题约再多 1 秒，换的是引用可核对。

## License

MIT
