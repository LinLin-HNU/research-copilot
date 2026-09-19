"""
PDF 论文解析模块
使用 PyMuPDF 提取文本，按学术论文章节结构分割
"""
from collections import Counter
import math
import re

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


def _normalise_margin_line(text: str) -> str:
    text = re.sub(r"\d+", "#", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_bytes(data: bytes) -> tuple[str, list[int]]:
    """Extract PDF text while removing repeated top and bottom margin noise."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[list[tuple[str, bool]]] = []
    margin_counts: Counter[str] = Counter()

    try:
        for page in doc:
            page_height = page.rect.height or 1
            page_lines: list[tuple[str, bool]] = []
            page_margin_lines = set()
            for block in page.get_text("blocks"):
                x0, y0, _x1, y1, block_text, _block_no, block_type = block
                if block_type != 0:
                    continue
                in_margin = y0 <= page_height * 0.12 or y1 >= page_height * 0.88
                for line in block_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    page_lines.append((line, in_margin))
                    if in_margin:
                        normalised = _normalise_margin_line(line)
                        if len(normalised) >= 3:
                            page_margin_lines.add(normalised)
                page_lines.append(("", False))
            margin_counts.update(page_margin_lines)
            pages.append(page_lines)
    finally:
        doc.close()

    repeated_threshold = max(3, math.ceil(len(pages) * 0.5))
    repeated_margin_lines = {
        line for line, count in margin_counts.items() if count >= repeated_threshold
    }
    all_lines: list[str] = []
    line_pages: list[int] = []
    for page_number, page_lines in enumerate(pages, start=1):
        for line, in_margin in page_lines:
            if in_margin and _normalise_margin_line(line) in repeated_margin_lines:
                continue
            all_lines.append(line)
            line_pages.append(page_number)
    return "\n".join(all_lines), line_pages


def _canonical_section_key(section: str, occurrence: int) -> str:
    return f"{section}@@{occurrence}"


def _section_from_key(section_key: str) -> str:
    return section_key.rsplit("@@", 1)[0]


def _heading_keyword(text: str) -> str | None:
    candidate_text = re.sub(
        r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)]?\s+", "", text.strip()
    )
    if candidate_text.endswith((".", ";", ",")):
        return None
    candidate = candidate_text.lower().rstrip(":")
    if len(candidate) > 120:
        return None
    for keyword in SECTION_KEYWORDS:
        if candidate == keyword or candidate.startswith(f"{keyword} "):
            return keyword
    return None


def detect_sections(
    text: str,
    line_pages: list[int] | None = None,
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """
    按章节关键词分割论文文本。
    返回: ( {section_name: content}, {section_name: (start_page, end_page)} )
    """
    lines = text.split("\n")
    sections: dict[str, str] = {}
    section_pages: dict[str, list[int]] = {}
    current_section = "preamble"
    current_lines = []
    current_line_pages: list[int] = []
    occurrences: Counter[str] = Counter()

    def flush():
        first = 0
        last = len(current_lines)
        while first < last and not current_lines[first].strip():
            first += 1
        while last > first and not current_lines[last - 1].strip():
            last -= 1
        content_lines = current_lines[first:last]
        if content_lines:
            occurrences[current_section] += 1
            key = _canonical_section_key(current_section, occurrences[current_section])
            sections[key] = "\n".join(content_lines)
            section_pages[key] = current_line_pages[first:last]

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            current_line_pages.append(line_pages[idx] if line_pages else 0)
            continue

        matched = _heading_keyword(stripped)

        if matched:                         #判断是否找到了新章节，如果当前matched不为None，说明当前行是一个新的章节标题
            flush()
            current_section = matched
            current_lines = [stripped]
            current_line_pages = [line_pages[idx] if line_pages else 0]
        else:
            current_lines.append(line)
            current_line_pages.append(line_pages[idx] if line_pages else 0)

    flush()
    return sections, section_pages


def chunk_sections(
    sections: dict[str, str],
    section_pages: dict[str, list[int]] | None = None,
    max_chars: int = 1500,
) -> list[dict]:
    """
    将章节分割为适合 RAG 的块。
    每块不超过 max_chars 字符，尽量保持段落完整，按照双换行符\n\n进行切割。
    返回: [{"section": str, "content": str, "page_start": int, "page_end": int}, ...]
    """
    chunks = []             # chunk整体存的是一个如"abstract"这样的章节，其中有多个元素，元素是字典，键是”abstract"，值是一个或多个段落的内容

    for section_key, content in sections.items():
        pages = (section_pages or {}).get(section_key, [0] * len(content.split("\n")))
        lines = content.split("\n")
        if len(pages) != len(lines):
            pages = [0] * len(lines)
        current_lines: list[str] = []
        current_pages: list[int] = []

        def flush_chunk() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                known_pages = [page for page in current_pages if page]
                chunks.append({
                    "section": _section_from_key(section_key),
                    "content": body,
                    "page_start": min(known_pages) if known_pages else 0,
                    "page_end": max(known_pages) if known_pages else 0,
                })

        for line, page in zip(lines, pages):
            projected_size = len("\n".join(current_lines + [line]))
            if current_lines and projected_size > max_chars:
                flush_chunk()
                current_lines = []
                current_pages = []
            current_lines.append(line)
            current_pages.append(page)
        flush_chunk()
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
