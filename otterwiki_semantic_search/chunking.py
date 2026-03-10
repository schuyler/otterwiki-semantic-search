"""Page content chunking for vector embedding."""

import re

from otterwiki_semantic_search.frontmatter import parse_frontmatter

TARGET_WORDS = 150
OVERLAP_WORDS = 35
MIN_CHUNK_WORDS = 300  # Pages below this are a single chunk


def _split_sentences(text):
    """Split text on sentence boundaries."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n', text)
    return [p for p in parts if p.strip()]


def _word_count(text):
    return len(text.split())


def chunk_page(pagepath, content):
    """Split a page into overlapping chunks for embedding.

    Returns list of dicts with keys: id, text, metadata.
    """
    frontmatter, body = parse_frontmatter(content)
    body = body.strip()

    if not body:
        return []

    # Build metadata from frontmatter
    meta = {"page_path": pagepath}
    if frontmatter:
        if "category" in frontmatter:
            meta["category"] = str(frontmatter["category"])
        if "tags" in frontmatter:
            tags = frontmatter["tags"]
            if isinstance(tags, list):
                meta["tags"] = ", ".join(str(t) for t in tags)
            else:
                meta["tags"] = str(tags)
        if "title" in frontmatter:
            meta["title"] = str(frontmatter["title"])

    # Short pages: single chunk
    if _word_count(body) < MIN_CHUNK_WORDS:
        return [
            {
                "id": f"{pagepath}::chunk_0",
                "text": body,
                "metadata": {**meta, "chunk_index": 0},
            }
        ]

    # Split on paragraph boundaries
    paragraphs = re.split(r'\n\n+', body)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_text_parts = []

    for para in paragraphs:
        para_wc = _word_count(para)

        # If a single paragraph is too large, split on sentences
        if para_wc > TARGET_WORDS * 2:
            # Flush current accumulation first
            if current_text_parts:
                chunks.append("\n\n".join(current_text_parts))
                current_text_parts = []

            sentences = _split_sentences(para)
            sent_acc = []
            sent_wc = 0
            for sent in sentences:
                sw = _word_count(sent)
                if sent_wc + sw > TARGET_WORDS and sent_acc:
                    chunks.append(" ".join(sent_acc))
                    sent_acc = []
                    sent_wc = 0
                sent_acc.append(sent)
                sent_wc += sw
            if sent_acc:
                chunks.append(" ".join(sent_acc))
            continue

        word_total = sum(_word_count(w) for w in current_text_parts) + para_wc

        if word_total > TARGET_WORDS and current_text_parts:
            chunks.append("\n\n".join(current_text_parts))
            current_text_parts = []

        current_text_parts.append(para)

    if current_text_parts:
        chunks.append("\n\n".join(current_text_parts))

    # Add overlap between adjacent chunks (always capped to OVERLAP_WORDS)
    if len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            overlap_words = prev_words[-OVERLAP_WORDS:]
            overlapped.append(" ".join(overlap_words) + "\n\n" + chunks[i])
        chunks = overlapped

    return [
        {
            "id": f"{pagepath}::chunk_{i}",
            "text": text,
            "metadata": {**meta, "chunk_index": i},
        }
        for i, text in enumerate(chunks)
    ]
