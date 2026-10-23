"""Markdown-aware splitting so formatting survives a rewrite.

Only plain prose paragraphs are rewritten.  Kept exactly as written:
front matter, headings, fenced/indented code, tables, block quotes,
HTML blocks, horizontal rules, list blocks, link-reference definitions.
Inside prose, inline code, links, images and URLs are passed to the
rewriter as protected terms so they are never altered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_LIST = re.compile(r"^\s{0,3}([-*+]|\d+[.)])\s")
_RULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_TABLE = re.compile(r"^\s*\|")
_QUOTE = re.compile(r"^\s{0,3}>")
_HTML = re.compile(r"^\s{0,3}<[A-Za-z!/]")
_REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s")
_INLINE_PROTECT = re.compile(r"`[^`]+`|!?\[[^\]]*\]\([^)]*\)|https?://\S+|\*\*|__")


@dataclass
class Block:
    text: str
    prose: bool


def split_blocks(md: str) -> list[Block]:
    """Split Markdown into prose and keep-as-is blocks.  Joining every
    block's text with ``"\\n\\n"`` gives back the paragraph structure."""
    lines = md.replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    i = 0
    if lines and lines[0].strip() == "---":  # YAML front matter
        j = next((k for k in range(1, len(lines)) if lines[k].strip() in ("---", "...")), None)
        if j is not None:
            blocks.append(Block("\n".join(lines[: j + 1]), False))
            i = j + 1
    para: list[str] = []

    def flush() -> None:
        if para:
            text = "\n".join(para)
            blocks.append(Block(text, _is_prose(para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            flush()
            fence = _FENCE.match(line).group(1)
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(fence):
                j += 1
            blocks.append(Block("\n".join(lines[i : j + 1]), False))
            i = j + 1
            continue
        if not line.strip():
            flush()
        elif _HEADING.match(line) or _RULE.match(line):
            flush()
            blocks.append(Block(line, False))
        else:
            para.append(line)
        i += 1
    flush()
    return blocks


def _is_prose(lines: list[str]) -> bool:
    first = lines[0]
    if first.startswith(("    ", "\t")):
        return False  # indented code
    return not any(p.match(first) for p in (_LIST, _TABLE, _QUOTE, _HTML, _REFDEF))


def protected_inline(text: str) -> list[str]:
    """Inline code, links, images and URLs that must stay untouched."""
    return list(dict.fromkeys(m.group(0) for m in _INLINE_PROTECT.finditer(text) if len(m.group(0)) > 2))
