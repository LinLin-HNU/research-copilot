# G0 冻结记录（2026-10-07）

开发工作树：`experiment/ablation-20261005`，起点提交 `3aeaefdc6ec9a3ca4ae44b46710aedd003b90e67`。当前工作树还有先前的文档与 G0 校验文件改动；G1 只增改自己需要的文件，不覆盖这些改动。

用户已认可 Informer、Autoformer、PatchTST、系统综述四道优先评审题的表述和一轮 8–16 处答案核心引用核查。决策轨迹复核是额外的简短工作量。G0 协议以 [`WORKFLOW_G0_PROTOCOL.md`](WORKFLOW_G0_PROTOCOL.md) 为准；不根据实验分数改题、改评分口径或读取保留题调参。

离线结构校验：12 题，其中论文 `dev` 8、论文 `reserve` 2、控制题 2；9 篇论文。综述 PDF 39 页，选定索引 104 块，页码范围 1–39。校验报告明确为结构检查，网络调用 0，不代表答案质量。

| 私有/来源文件 | SHA-256（仅公开摘要，不公开内容或会话 ID） |
| --- | --- |
| `private/workflow_g0_20261006/cases.json` | `3b43042f60707a1a91d72991abd80ee76b317021c613f2ee36b3a844be89b089` |
| 主仓库 `benchmark/drafts/timeseries_20260922_curated_release2/ai_reviewed_dev_cases.json` | `f0694132b84cecee4f2f4a122f56b8d23693a667c41fb7f08afca53bc7c0cebe` |
| `private/workflow_g0_20261006/survey_source.private.json` | `7ab7b67f12f90184c63caa0cb7f44e8eb64ef9df44295663b41f6f8d02c4e1a6` |
| 综述 PDF | `2b475d13cc1bf25a4c7de6cd5faa0af7a24383f1c1bdaaa463ac8be3f5e88c80` |

B0 上限：最多 2 次检索或展开动作、3 次模型调用、8 个唯一证据块、6000 证据字符；第一次证据约 4000 字；单次模型超时 60 秒、整个请求 180 秒。三次模型输出上限依次为 400、400、1200 token。正式实验前仍需核实端点、归档向量库来源和实际 API usage；字符限制不等于费用限制。

G1 实现与离线验收可以开始。真实论文缺口识别准确性必须由后续逐题轨迹与人工复核判断，离线用例只证明程序分支和限制条件运行。
