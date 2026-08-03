"""
PDF 论文解析模块
使用 PyMuPDF 提取文本，按学术论文章节结构分割
"""
import fitz

# 常见学术论文章节关键词（按优先级排序）
SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "method",
    "approach",
    "proposed method",
    "proposed approach",
    "system design",
    "architecture",
    "experiment",
    "experimental setup",
    "experimental results",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "future work",
    "references",
    "appendix",
]


def extract_text_from_bytes(data: bytes) -> tuple[str, list[int]]:
    """从 PDF 提取全文，并返回每一行所在的页码（从 1 开始）"""
    doc = fitz.open(stream=data, filetype="pdf")
    all_lines = []
    line_pages = []
    for page_idx, page in enumerate(doc):
        for line in page.get_text().split("\n"):
            all_lines.append(line)
            line_pages.append(page_idx + 1)
    doc.close()
    return "\n".join(all_lines), line_pages


def detect_sections(text: str, line_pages: list[int] | None = None) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    """
    按章节关键词分割论文文本。
    返回: ( {section_name: content}, {section_name: (start_page, end_page)} )
    """
    lines = text.split("\n")
    sections = {}
    section_pages = {}
    current_section = "preamble"
    current_lines = []
    section_start_line = 0

    def flush():
        content = "\n".join(current_lines).strip()
        if content:
            sections[current_section] = content
            if line_pages is not None and current_lines:
                start = line_pages[section_start_line]
                end = line_pages[section_start_line + len(current_lines) - 1]
                section_pages[current_section] = (start, end)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)      #如果是空行也依然保留，因为后续chunk_sections函数会根据双换行符\n\n来切分段落
            continue                        #跳过当前判断，因为当前是按行遍历，空行不可能是一个章节标题，所以直接跳过本轮循环，进入下一行

        lower = stripped.lower().rstrip(".:")
        matched = None
        for keyword in SECTION_KEYWORDS:
            # 匹配章节标题：行首关键词，且行较短（标题特征）
            """ ！！！！！    关于这个匹配规则，如果论文的标题不按常理怎么办，keyword这些关键词是包含在标题中，而不是在标题首部或只有一个简单的像“abstract”"""
            if lower.startswith(keyword) or lower == keyword:
                if len(stripped) < 100:  # 标题一般不超过100字符
                    matched = keyword
                    break

        if matched:                         #判断是否找到了新章节，如果当前matched不为None，说明当前行是一个新的章节标题
            flush()
            current_section = matched       #更新当前章节标识：把当前章节名换成新匹配到的关键词
            current_lines = [stripped]      #重置缓存：清空之间的行缓存，并将当前这个标题的行作为新章节的第一行加入缓存
            section_start_line = idx        #记录新章节起始行号，用于推算页码范围
        else:
            current_lines.append(line)

    flush()
    return sections, section_pages


def chunk_sections(
    sections: dict[str, str],
    section_pages: dict[str, tuple[int, int]] | None = None,
    max_chars: int = 1500,
) -> list[dict]:
    """
    将章节分割为适合 RAG 的块。
    每块不超过 max_chars 字符，尽量保持段落完整，按照双换行符\n\n进行切割。
    返回: [{"section": str, "content": str, "page_start": int, "page_end": int}, ...]
    """
    chunks = []             # chunk整体存的是一个如"abstract"这样的章节，其中有多个元素，元素是字典，键是”abstract"，值是一个或多个段落的内容

    def with_page(chunk: dict) -> dict:
        start = end = 0
        if section_pages:
            page_range = section_pages.get(chunk["section"])
            if page_range:
                start, end = page_range
        chunk["page_start"] = start
        chunk["page_end"] = end
        return chunk

    for section_name, content in sections.items():
        if len(content) <= max_chars:
            chunks.append(with_page({"section": section_name, "content": content}))
        else:
            # 按段落拆分
            paragraphs = content.split("\n\n")      #按照双换行符\n\n切分成多个段落并存入列表paragraphs
            current = ""
            for para in paragraphs:
                if len(current) + len(para) <= max_chars:
                    current += para + "\n\n"
                else:
                    if current.strip():
                        chunks.append(with_page({"section": section_name, "content": current.strip()}))    #如果超过字符限制，就把它加入列表chunk当中，作为chunk中的一个元素
                    current = para + "\n\n"
            if current.strip():
                chunks.append(with_page({"section": section_name, "content": current.strip()}))
    return chunks


def guess_title(text: str) -> str:
    """从文本首部猜测论文标题（取前几行中最可能是标题的行）"""
    lines = [l.strip() for l in text.split("\n") if l.strip()][:10]
    for line in lines:
        # 标题通常是前几行中较长、不带关键词的行
        if len(line) > 20 and not any(
            kw in line.lower() for kw in ["abstract", "introduction", "copyright", "arxiv"]
        ):
            return line[:100]
    return "未命名论文"
