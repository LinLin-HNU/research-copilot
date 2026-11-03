"""Paper answer caps shared by the workflow and newly frozen pilots.

These bound complete serialized model output, not just visible prose. Source,
tool and call budgets remain separate. Archived manifests are not rewritten.
即集中定义不同讲解模式（全文/RAG）深度的输出上限
"""
ADVANCED_PAPER_OUTPUT_TOKENS = 6400
FULL_PAPER_OUTPUT_LIMITS = {"concise": 1200, "beginner": 2600,
                            "advanced": ADVANCED_PAPER_OUTPUT_TOKENS}
RAG_PAPER_OUTPUT_LIMITS = {"concise": 1200, "beginner": 2400,
                           "advanced": ADVANCED_PAPER_OUTPUT_TOKENS}
