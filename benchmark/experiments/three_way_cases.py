"""Draft development-acceptance questions, not expert gold or holdout data."""
CASES = [
    {"id": "tw-01", "batch": 1, "paper": "informer", "question": "请讲解 Informer 要解决的主要问题及三项核心贡献，并解释每项贡献解决什么问题。", "review_focus": "三项贡献的作用；复杂度与适用条件不要混淆。"},
    {"id": "tw-02", "batch": 1, "paper": "autoformer", "question": "请深入解释 Autoformer 中序列分解与 Auto-Correlation 如何协作，按输入、编码器、解码器、最终预测的顺序讲清趋势与季节信息怎样流动。", "review_focus": "方法链路与趋势/季节分工；不要只列术语。"},
    {"id": "tw-03", "batch": 1, "paper": "patchtst", "question": "请详细解释 PatchTST 怎样把原始多变量序列变成 patch token，以及 channel-independence 的含义、参数共享方式和作用。", "review_focus": "patch表示、通道独立和参数共享；不能误称显式跨变量注意力。"},
    {"id": "tw-04", "batch": 1, "paper": "fedformer", "question": "请详细解释 FEDformer 的序列分解和 Frequency Enhanced Block 分别负责什么，以及二者如何组成预测流程；请区分频域表示处理与简单滤波。", "review_focus": "分解与频域模块分工；解释属于原文还是教学性推断。"},
    {"id": "tw-05", "batch": 2, "paper": "timesfm", "question": "请按输入、patch表示、decoder-only骨干、预测输出的顺序解释 TimesFM，并说明论文中的零样本预测意味着什么、不能保证什么。", "review_focus": "训练/预测链路；零样本的条件和验证边界。"},
    {"id": "tw-06", "batch": 2, "paper": "survey", "question": "请依据这篇系统综述解释点预测与概率预测的差别，以及各自代表性的评价指标；说明这些指标评价的对象，不要只列缩写。", "review_focus": "预测对象与评分对象；不把概率预测指标混作单值误差。"},
    {"id": "tw-07", "batch": 2, "paper": "informer", "question": "有人说 Informer 的生成式解码器在预测时逐步输入真实未来值。这个说法是否符合论文？请核查并解释其预测输入和输出方式；证据不能确认的部分请说明。", "review_focus": "核查错误前提；不要顺着错误前提编造。"},
    {"id": "tw-08", "batch": 2, "paper": "fedformer", "question": "FEDformer 作者明确写出的局限有哪些？你认为它还有哪些值得研究的改进方向？请把作者原文、基于方法的推断及你的建议分开；找不到作者表述时不要代替作者承认局限。", "review_focus": "作者局限与读者推断的归属；原文不能确认就明确说未确认。"},
]
TITLE_CUES = {"informer": "informer", "autoformer": "autoformer", "fedformer": "fedformer",
              "patchtst": "a time series is worth 64 words", "timesfm": "decoder-only foundation",
              "survey": "systematic review"}
