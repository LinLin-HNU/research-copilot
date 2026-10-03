"""Create human-curated, source-validated dev-question drafts for nine uploaded papers.

Unlike draft_questions.py, this uses paper-specific concepts selected by a reviewer.
It validates that each expected term exists in the locally parsed corpus and records
the exact section/page excerpt used for later human gold-label review.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_store import get_rag

SPECS = {
"inf": [("A1","论文提出的 LSTF 要解决什么问题？Informer 的三项核心设计分别是什么？",["abstract"],"LSTF"),("A1","ProbSparse self-attention 的设计动机和作用是什么？",["introduction"],"ProbSparse"),("A1","Informer 如何通过 self-attention distilling 处理长序列？",["introduction"],"distilling"),("A1","论文使用哪些数据集评估 Informer？以 ETTh1 为例说明实验结论。",["experiment","evaluation","results"],"ETTh1"),("A1","generative style decoder 与逐步预测方式有何差异？",["introduction","evaluation"],"generative style decoder"),("A2","论文讨论的长序列预测挑战或局限是什么？",["discussion","conclusion"],"long sequence"),("C","基于 ProbSparse 的设计，怎样验证它在极长历史窗口下是否仍有效？",["introduction"],"ProbSparse"),("C","若要改进 Informer 对多变量关系的建模，你会提出什么可检验的方向？",["evaluation","experiment"],"multivariate")],
"aut": [("A1","Autoformer 的核心贡献是什么？它解决了传统 Transformer 的什么问题？",["abstract"],"Autoformer"),("A1","论文如何把 series decomposition 融入网络结构？",["architecture"],"series decomposition"),("A1","Auto-Correlation mechanism 如何发现并聚合周期性依赖？",["architecture"],"Auto-Correlation"),("A1","Autoformer 的 progressive decomposition 在模型中起什么作用？",["architecture"],"progressive decomposition"),("A1","论文在哪些数据集或预测设置上验证 Autoformer？",["experiment"],"ETT"),("A2","传统 self-attention 用于长序列预测时有哪些限制？",["introduction"],"self-attention"),("C","若周期性不稳定，如何扩展 Auto-Correlation 的设计并通过实验验证？",["architecture"],"periodicity"),("C","如何设计消融实验，区分 series decomposition 与 Auto-Correlation 分别带来的收益？",["architecture"],"Auto-Correlation")],
"fed": [("A1","FEDformer 要解决长时序预测的哪些问题？",["abstract"],"FEDformer"),("A1","论文中的 seasonal-trend decomposition 分别负责捕捉什么信息？",["introduction"],"seasonal-trend decomposition"),("A1","Frequency Enhanced Block 如何利用 Fourier 变换？",["architecture"],"Fourier"),("A1","FEDformer 为什么强调线性复杂度？",["abstract","architecture"],"linear complexity"),("A1","论文如何将频域表示用于长时序预测？",["architecture"],"frequency"),("A2","标准 Transformer 用于该任务时为何难以捕捉全局趋势？",["introduction"],"global view"),("C","如何比较 Fourier 与 Wavelet 组件分别对不同周期数据的影响？",["architecture"],"Wavelet"),("C","如果数据的趋势突变，怎样修改 seasonal-trend decomposition 并验证？",["introduction"],"seasonal-trend decomposition")],
"pat": [("A1","PatchTST 的核心设计和主要贡献是什么？",["abstract"],"PatchTST"),("A1","论文中的 patching 如何改变时间序列输入表示？",["proposed method","method"],"patching"),("A1","channel-independence 假设是什么，为什么有用？",["method","proposed method"],"channel-independence"),("A1","PatchTST 如何处理多变量时间序列？",["method","proposed method"],"multivariate"),("A1","论文在长时序预测实验中如何评价 PatchTST？",["results"],"ETTh1"),("A2","论文指出 Transformer-based 预测器面临哪些限制？",["conclusion","introduction"],"Transformer-based"),("C","如果变量间强相关，怎样检验 channel-independence 是否成为限制？",["method","proposed method"],"channel-independence"),("C","如何调整 patch length，并设计实验衡量其影响？",["proposed method","method"],"patch length")],
"itr": [("A1","iTransformer 的基本思想是什么？它为何反转传统 Transformer 的维度处理方式？",["abstract"],"iTransformer"),("A1","论文中的 variate tokens 与传统 temporal tokens 有何不同？",["introduction"],"variate tokens"),("A1","iTransformer 的 attention 和 feed-forward network 分别建模什么？",["abstract","introduction"],"feed-forward"),("A1","论文如何处理更长的 lookback windows？",["abstract","results"],"lookback windows"),("A1","iTransformer 在多变量数据上的表现由什么机制支撑？",["abstract","introduction"],"multivariate correlations"),("A2","传统 temporal tokens 为什么可能造成无意义的 attention maps？",["introduction"],"attention maps"),("C","若变量数量大幅增加，如何评估 iTransformer 的可扩展性？",["abstract","results"],"variates"),("C","如何通过消融实验检验 inverted dimensions 是否是性能来源？",["introduction"],"inverted dimensions")],
"tll": [("A1","Time-LLM 的目标是什么？它如何利用冻结的语言模型进行预测？",["abstract"],"Time-LLM"),("A1","Time-LLM 的 reprogramming 机制如何对齐时间序列与语言模型？",["approach"],"reprogramming"),("A1","text prototypes 在框架中发挥什么作用？",["approach"],"text prototypes"),("A1","Prompt-as-Prefix 如何丰富模型输入上下文？",["approach"],"Prompt-as-Prefix"),("A1","论文如何验证 Time-LLM 的 few-shot 或 zero-shot 能力？",["evaluation","conclusion"],"few-shot"),("A2","时间序列数据稀疏为何会限制 foundation models 的发展？",["introduction"],"data sparsity"),("C","如何检验不同 LLM backbone 对 Time-LLM 结果的影响？",["approach","evaluation"],"LLM"),("C","若文本原型与序列模式不匹配，怎样改进 reprogramming 并验证？",["approach"],"reprogramming")],
"tsn": [("A1","TimesNet 的 TimesBlock 要解决什么时间变化建模问题？",["abstract"],"TimesBlock"),("A1","论文如何根据 multiple periods 将一维序列转换为二维表示？",["architecture"],"multiple periods"),("A1","intra-period 与 inter-period variations 分别是什么？",["abstract","architecture"],"interperiod"),("A1","Inception block 在 TimesBlock 中承担什么作用？",["architecture"],"inception"),("A1","TimesNet 在哪些时间序列任务上进行了验证？",["abstract","results"],"imputation"),("A2","从论文结论看，TimesNet 的适用范围或仍待解决问题是什么？",["conclusion"],"TimesNet"),("C","如何验证二维变换对非周期序列是否仍有益？",["architecture"],"periodicity"),("C","若只做异常检测，应怎样调整 TimesBlock 并设计对照实验？",["abstract","results"],"anomaly")],
"tfm": [("A1","TimesFM 的预训练目标和零样本预测定位是什么？",["abstract"],"zero-shot"),("A1","论文为何采用 decoder style attention 与 input patching？",["abstract","introduction"],"decoder style attention"),("A1","模型如何适应不同历史长度和预测长度？",["abstract"],"history lengths"),("A1","TimesFM 的训练语料或时间序列语料库有什么作用？",["abstract","introduction"],"time-series corpus"),("A1","论文如何将 TimesFM 的零样本表现与监督模型比较？",["abstract","introduction"],"supervised"),("A2","零样本基础模型在特定数据集上可能有哪些局限或验证风险？",["conclusion","introduction"],"zero-shot"),("C","如何设计跨频率数据集实验，检验 TimesFM 的泛化能力？",["abstract"],"temporal granularities"),("C","若要引入外部协变量，怎样扩展 decoder-only 模型并验证？",["introduction","abstract"],"decoder")],
"chr": [("A1","Chronos 的核心方法是什么？它如何把时间序列变成语言模型可处理的表示？",["abstract"],"Chronos"),("A1","scaling 和 quantization 在 Chronos tokenization 中分别做什么？",["abstract","background"],"quantization"),("A1","Chronos 为什么采用 fixed vocabulary 和 cross-entropy loss？",["abstract","background"],"fixed vocabulary"),("A1","论文使用哪些规模的 T5 模型训练 Chronos？",["abstract"],"T5"),("A1","Chronos 如何在未见数据集上评估 zero-shot 预测能力？",["evaluation","abstract"],"zero-shot"),("A2","论文中 synthetic dataset 的使用可能带来哪些泛化风险？",["abstract","discussion"],"synthetic dataset"),("C","如何比较不同 quantization 粒度对预测精度和不确定性的影响？",["abstract","background"],"quantization"),("C","如果遇到分布漂移，怎样扩展 Chronos 的 tokenization 并验证？",["discussion","conclusion"],"tokenization")],
}


def find_source(chunks, term, preferred):
    # PDF extraction may split a hyphenated phrase across a line (for example,
    # "patched-\ndecoder"). Compare a punctuation/whitespace-insensitive form.
    normalized_term = re.sub(r"[^a-z0-9]+", "", term.lower())
    matching = [
        row for row in chunks
        if normalized_term in re.sub(r"[^a-z0-9]+", "", row["content"].lower())
    ]
    if not matching:
        raise ValueError(f"Term not found in corpus: {term}")
    for section in preferred:
        rows = [row for row in matching if row.get("section", "").lower() == section]
        if rows:
            return max(rows, key=lambda row: len(row["content"]))
    return max(matching, key=lambda row: len(row["content"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sessions = json.loads(args.sessions.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    rag = get_rag()
    all_cases, all_review = [], []
    for session in sessions:
        prefix, thread_id = session["prefix"], session["thread_id"]
        if prefix not in SPECS:
            raise ValueError(f"No curated specification for {prefix}")
        chunks = rag.list_chunks(thread_id)
        cases = []
        for number, (category, question, preferred_sections, term) in enumerate(SPECS[prefix], 1):
            source = find_source(chunks, term, preferred_sections)
            section = source["section"].lower()
            case_id = f"{prefix}-{category.lower()}-{number:02d}"
            case = {"id": case_id, "question": question, "expected_category": category,
                    "expected_sections": [section], "expected_terms": [term], "thread_id": thread_id, "split": "dev"}
            cases.append(case)
            all_review.append({"paper_name": session["paper_name"], "id": case_id, "question": question,
                               "category": category, "expected_section": section, "expected_term": term,
                               "pdf_pages": f"{source.get('page_start',0)}-{source.get('page_end',0)}",
                               "candidate_excerpt": source["content"][:1200], "review_status": "needs_human_gold"})
        for number, question in ((9, "什么是时间序列预测中的训练集、验证集和测试集？它们分别用于什么？"), (10, "MAE 和 MSE 分别衡量什么？在比较预测模型时应如何理解它们？")):
            cases.append({"id": f"{prefix}-b-{number:02d}", "question": question, "expected_category": "B", "expected_sections": [], "expected_terms": [], "thread_id": thread_id, "split": "dev"})
        if len(cases) != 10:
            raise AssertionError("Each paper must have 10 cases")
        (args.output/f"{prefix}.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        all_cases.extend(cases)
    (args.output/"all_drafts.json").write_text(json.dumps(all_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output/"review_packets.json").write_text(json.dumps(all_review, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output/"manifest.json").write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"papers": len(sessions), "draft_cases": len(all_cases), "review_rows": len(all_review)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
