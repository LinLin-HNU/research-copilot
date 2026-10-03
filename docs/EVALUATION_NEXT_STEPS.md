# 评测修订：先固定证据与口径

本次保留 A1/A2/B/C 路由设计，未修改检索与生成逻辑，未调用付费 API。

旧 Judge 用统一候选片段（约 1200 字）核验全部答案，已发现将实际上下文中的
Traffic x22 判为幻觉的案例。新脚本 judge_evidence_v2.py 使用每个答案保存的完整
evidence，指标为 evidence_faithfulness（对检索上下文的忠实性）、citation_support
（对已有引用的支持）、completeness（对共同参考要点的覆盖）。它不测整篇论文事实真伪。

同题答案的参考要点必须相同；实际生成证据可以不同，这正是要检查的上下文。
答案格式可能泄露系统特征，因此仅隐藏显式版本标记，不能称绝对盲法。
程序会核验裁判给出的引文是否为输入原文子串；核验不通过、证据不足、缺少轮次、
出现 API 错误时，相关分数保留 NA。NA 的分母及同题配对数必须一起报告。
程序保留模型原始分数及覆盖原因，连续三次失败停止，避免持续付费失败。
已有旧评审文件保留，不修改其分数或与新协议混合汇总。

## 当前离线产物

private/judge_evidence_v2_old_answers/ 保存 72 对旧答案的完整上下文包。
private/judge_evidence_v2_smoke_old_answers/ 保存固定种子抽取的 2 对（4 个）旧答案。
两者仍是修复路由前的答案，只适合核验新裁判流程，不能证明修改后 V1 的质量。

## 可选的 4 次付费冒烟

在项目根目录、已配置 Python 环境执行以下单行命令。无需启动 Web/Docker。

```powershell
python -u benchmark/experiments/judge_evidence_v2.py run --packets benchmark/experiments/private/judge_evidence_v2_smoke_old_answers/packets.json --passes 1 --confirm-paid --output benchmark/experiments/private/judge_evidence_v2_smoke_run
python benchmark/experiments/judge_evidence_v2.py score --run benchmark/experiments/private/judge_evidence_v2_smoke_run --packets benchmark/experiments/private/judge_evidence_v2_smoke_old_answers --output benchmark/experiments/private/judge_evidence_v2_smoke_scored.json
```

新生成的 V1 答案需先与 V0 通过 make_blind_sheet.py 生成新的 mapping，再运行：

```powershell
python benchmark/experiments/judge_evidence_v2.py prepare --mapping 新盲审目录/mapping.private.json --output 新证据包目录
```

## 待用户决定的架构方向

A1/A2/C 当前既影响检索查询与章节偏好，也影响回答要求；路由错误会影响下游，
但不是必然导致答案错误。可以之后比较“只用论文相关/通用知识两类决定是否检索，
其余类别仅影响写作要求”，以及当前四类策略。应固定证据预算，补真实 B 类题，
分别测漏检、无必要检索、答案质量和 token。现有 72 题没有 B 类，不足以衡量节省
token 的收益与误判代价；也不能把开发集 100% 当成独立测试集泛化结论。

暂不需要重跑全量。先核验少量 Judge 输出，等路由设计确定后再生成小样本新答案。
