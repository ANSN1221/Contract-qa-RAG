# 种子题集（Seed Set）

15 条人工标注的题，用于测 Recall@5。**不参与训练，只用于评测。**

## 格式

`seeds.jsonl`，每行一条：

```json
{"id": "Q01", "question": "合同总价是多少？", "relevant_clause_ids": ["DOC002-C0001"], "category": "fact"}
```

| 字段 | 规则 |
| --- | --- |
| `id` | Q01–Q50 |
| `question` | 用真实用户口吻，不要照抄条款原文 |
| `relevant_clause_ids` | 可多个；跨条款题填 2–3 个，命中任一即算召回成功；不可答题填 `[]` |
| `category` | `fact`（事实，17 条）/ `cross`（跨条款，17 条）/ `unanswerable`（不可答，16 条） |

## 标注规则

1. **只记 clause_id，不记页码**。页码属于引用正确性，另有验收脚本测。
2. **答案必须能在该条款原文里直接读到**，不能靠推理。
3. **跨条款题**的多个 id 必须是「都回答了问题」的条款，不是「相关但没答」的。
4. **不可答题**要求语料里确实没有答案（如「项目经理手机号」），而不是「懒得找」。

## 辅助文件

- `candidates.csv`：569 条条款的清单（id / 文档 / 条款号 / 章节 / 页码 / 前 80 字），用 Excel 或 VS Code 打开浏览。
  重新生成：`uv run python scripts/build_candidates.py`

## 评分口径

```
Recall@5 = 前 5 条结果中至少命中一个 relevant_clause_id 的题数 / 总题数（不含 unanswerable）
```

宽松口径：跨条款题命中任一即算对。理由是这个指标衡量「用户能不能拿到答案」，不是「是否召回全部相关条款」。
