"""Fast PDF reading: text AND hyperlinks in a single pass with PyMuPDF.

Kept in its own module (standard library + PyMuPDF only) so it can run in worker processes:
PDF parsing is CPU work, and Python threads cannot run it in parallel. A worker process
imports only this file, never Streamlit or app.py.

    pdf_text_and_links(data, max_pages) -> (text, links)

Raises ImportError when PyMuPDF is not installed (the caller then falls back to
pdfplumber / pypdf) and ValueError when the PDF cannot be read.
"""

from __future__ import annotations


def available() -> bool:
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        return False
    return True


def pdf_text_and_links(data: bytes, max_pages: int = 20) -> tuple[str, list[str]]:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{type(e).__name__}: {str(e)[:120]}") from None
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError("PDF is password protected")
        # Expand ligatures ("ﬁ" -> "fi") so words such as "Firebase" are matched.
        flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
        texts: list[str] = []
        links: list[str] = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            # sort=True: top-to-bottom, left-to-right reading order (as pdfplumber does).
            texts.append(page.get_text("text", flags=flags, sort=True) or "")
            # Hyperlinks whose visible text is just "GitHub" only exist as link annotations.
            for link in page.get_links():
                uri = link.get("uri")
                if uri:
                    links.append(str(uri))
        return "\n".join(texts), links
    finally:
        doc.close()
