from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def strip_numbering(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^Appendix\s+[A-Z]\.\s*", "", text)
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text)


def escape_plain(text: str) -> str:
    return "".join(SPECIALS.get(ch, ch) for ch in text)


def convert_inline(text: str) -> str:
    placeholders: list[str] = []

    def stash_raw(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"@@RAW{len(placeholders) - 1}@@"

    def stash_code(match: re.Match[str]) -> str:
        raw = match.group(1)
        placeholders.append(r"\texttt{" + escape_plain(raw) + "}")
        return f"@@RAW{len(placeholders) - 1}@@"

    text = re.sub(r"\\cite[a-zA-Z]*\{[^}]+\}", stash_raw, text)
    text = re.sub(r"(?<!\$)\$[^$\n]+\$(?!\$)", stash_raw, text)
    text = re.sub(r"\\\(.*?\\\)", stash_raw, text)
    text = re.sub(r"`([^`]+)`", stash_code, text)
    text = escape_plain(text)

    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: r"\textbf{" + m.group(1) + "}",
        text,
    )
    text = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        lambda m: r"\emph{" + m.group(1) + "}",
        text,
    )

    for idx, value in enumerate(placeholders):
        text = text.replace(f"@@RAW{idx}@@", value)
    return text


def heading_command(level: int, title: str, in_appendix: bool) -> tuple[list[str], bool]:
    stripped = title.strip()
    lines: list[str] = []
    if stripped.lower().startswith("appendix ") and not in_appendix:
        lines.append(r"\appendix")
        in_appendix = True
    clean = convert_inline(strip_numbering(stripped))
    if level <= 2:
        lines.append(r"\section{" + clean + "}")
    elif level == 3:
        lines.append(r"\subsection{" + clean + "}")
    elif level == 4:
        lines.append(r"\subsubsection{" + clean + "}")
    else:
        lines.append(r"\paragraph{" + clean + "}")
    return lines, in_appendix


def close_list(open_list: str | None, out: list[str]) -> None:
    if open_list == "itemize":
        out.append(r"\end{itemize}")
    elif open_list == "enumerate":
        out.append(r"\end{enumerate}")


def convert_body(markdown: str) -> tuple[str, str]:
    lines = markdown.splitlines()
    title = "Untitled Paper"
    start = 0
    for idx, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            start = idx + 1
            break

    out: list[str] = []
    in_code = False
    code_lang = ""
    raw_latex_block = False
    open_list: str | None = None
    paragraph: list[str] = []
    in_abstract = False
    saw_appendix = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(convert_inline(" ".join(part.strip() for part in paragraph)))
            out.append("")
            paragraph = []

    def end_abstract_if_needed() -> None:
        nonlocal in_abstract
        if in_abstract:
            flush_paragraph()
            close_list(open_list, out)
            out.append(r"\end{abstract}")
            out.append("")
            in_abstract = False

    for raw in lines[start:]:
        line = raw.rstrip()

        fence = re.match(r"^```(\w*)", line)
        if fence:
            flush_paragraph()
            close_list(open_list, out)
            open_list = None
            if not in_code:
                code_lang = fence.group(1) or "text"
                raw_latex_block = code_lang.lower() in {"latex", "tex", "rawlatex"}
                if raw_latex_block:
                    pass
                elif code_lang == "text":
                    out.append(r"\begin{lstlisting}[breaklines=true]")
                else:
                    out.append(r"\begin{lstlisting}[language=" + code_lang + r",breaklines=true]")
                in_code = True
            else:
                if not raw_latex_block:
                    out.append(r"\end{lstlisting}")
                out.append("")
                in_code = False
                code_lang = ""
                raw_latex_block = False
            continue

        if in_code:
            out.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            close_list(open_list, out)
            open_list = None
            continue

        heading = re.match(r"^(#{2,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_list(open_list, out)
            open_list = None
            heading_text = heading.group(2).strip()
            if heading_text.lower() == "abstract":
                out.append(r"\begin{abstract}")
                in_abstract = True
            else:
                end_abstract_if_needed()
                commands, saw_appendix = heading_command(len(heading.group(1)), heading_text, saw_appendix)
                out.extend(commands)
                out.append("")
            continue

        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if bullet or ordered:
            flush_paragraph()
            wanted = "itemize" if bullet else "enumerate"
            if open_list != wanted:
                close_list(open_list, out)
                out.append(r"\begin{" + wanted + "}")
                open_list = wanted
            item_text = bullet.group(1) if bullet else ordered.group(1)
            out.append(r"\item " + convert_inline(item_text))
            continue

        if open_list:
            close_list(open_list, out)
            open_list = None
        paragraph.append(line)

    flush_paragraph()
    close_list(open_list, out)
    if in_abstract:
        out.append(r"\end{abstract}")
        out.append("")

    return title, "\n".join(out).strip() + "\n"


def write_main(title: str, body: str, output: Path) -> None:
    content = rf"""\documentclass[letterpaper]{{article}} % DO NOT CHANGE THIS
\usepackage[submission]{{aaai2027}}  % DO NOT CHANGE THIS
\usepackage[hyphens]{{url}}  % DO NOT CHANGE THIS
\usepackage{{graphicx}} % DO NOT CHANGE THIS
\urlstyle{{rm}} % DO NOT CHANGE THIS
\def\UrlFont{{\rm}}  % DO NOT CHANGE THIS
\usepackage{{natbib}}  % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\usepackage{{caption}} % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\frenchspacing  % DO NOT CHANGE THIS

\usepackage{{amsmath}}
\usepackage{{algorithm}}
\usepackage{{algorithmic}}
\usepackage{{newfloat}}
\usepackage{{listings}}
\usepackage{{booktabs}}
\DeclareCaptionStyle{{ruled}}{{labelfont=normalfont,labelsep=colon,strut=off}} % DO NOT CHANGE THIS
\lstset{{
  basicstyle={{\footnotesize\ttfamily}},
  numbers=left,
  numberstyle=\footnotesize,
  xleftmargin=2em,
  aboveskip=0pt,
  belowskip=0pt,
  showstringspaces=false,
  tabsize=2,
  breaklines=true
}}
\floatstyle{{ruled}}
\newfloat{{listing}}{{tb}}{{lst}}{{}}
\floatname{{listing}}{{Listing}}

\pdfinfo{{
/TemplateVersion (2027.1)
}}
\setcounter{{secnumdepth}}{{0}}

\title{{{convert_inline(title)}}}
\author{{Anonymous Authors}}
\affiliations{{}}

\begin{{document}}
\maketitle

{body}

\bibliography{{references}}

\end{{document}}
"""
    output.write_text(content, encoding="utf-8", newline="\n")


def export(root: Path) -> None:
    paper = root / "paper.md"
    template_dir = root / "templates" / "latex" / "aaai27"
    related_bib = root / "relatedwork" / "paper_list.bib"
    out_dir = root / "target" / "latex"

    if not paper.exists():
        raise FileNotFoundError(paper)
    if not template_dir.exists():
        raise FileNotFoundError(template_dir)
    if not related_bib.exists():
        raise FileNotFoundError(related_bib)

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in template_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)
    shutil.copy2(related_bib, out_dir / "references.bib")

    title, body = convert_body(paper.read_text(encoding="utf-8"))
    write_main(title, body, out_dir / "main.tex")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    export(args.root.resolve())


if __name__ == "__main__":
    main()
