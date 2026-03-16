"""Page content chunking for vector embedding."""

import re

from otterwiki_semantic_search.frontmatter import parse_frontmatter

TARGET_WORDS = 150
OVERLAP_WORDS = 35
MIN_CHUNK_WORDS = 300  # Pages below this are a single chunk
STUB_WORDS = 50
MAX_PREFIX_DEPTH = 3

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')


def _split_sentences(text):
    """Split text on sentence boundaries."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n', text)
    return [p for p in parts if p.strip()]


def _word_count(text):
    return len(text.split())


def _chunk_text(text: str) -> list:
    """Split text into chunks by paragraph/sentence boundaries. No overlap."""
    paragraphs = re.split(r'\n\n+', text)
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

    return chunks


def _split_into_sections(body):
    """Split body into list of dicts: {level, heading, text}.

    Level 0 = preamble (text before any heading).
    """
    lines = body.split('\n')
    sections = []
    current_level = 0
    current_heading = ""
    current_lines = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            # Save current section
            text = "\n".join(current_lines).strip()
            sections.append({
                "level": current_level,
                "heading": current_heading,
                "text": text,
            })
            current_level = len(m.group(1))
            current_heading = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Final section
    text = "\n".join(current_lines).strip()
    sections.append({
        "level": current_level,
        "heading": current_heading,
        "text": text,
    })

    return sections


def _build_header_stack(sections, page_title):
    """Annotate each section dict with section_path and section fields.

    Mutates sections in place, returns them.
    """
    # Stack of (level, title) tuples — does not include page_title entry
    stack = []

    for sec in sections:
        level = sec["level"]
        heading = sec["heading"]

        if level == 0:
            # Preamble
            sec["section"] = ""
            sec["section_path"] = page_title
        else:
            # Pop stack entries at same or deeper level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))

            # Build path: page_title + up to MAX_PREFIX_DEPTH levels from stack
            path_parts = [page_title] + [s[1] for s in stack[:MAX_PREFIX_DEPTH]]
            sec["section_path"] = " > ".join(path_parts)
            sec["section"] = heading

    return sections


def _merge_stub_sections(sections):
    """Merge stub sections (text < STUB_WORDS) forward or backward.

    Only non-preamble (level > 0) sections are candidates for merging.
    Returns new list of section dicts with stubs merged.
    """
    if not sections:
        return sections

    result = list(sections)

    # Forward pass: merge stubs into the next section (only if next section is substantial)
    i = 0
    while i < len(result) - 1:
        sec = result[i]
        next_sec = result[i + 1]
        if (sec["level"] > 0
                and _word_count(sec["text"]) < STUB_WORDS
                and _word_count(next_sec["text"]) >= STUB_WORDS):
            # Prepend text to next section, remove this section
            if sec["text"]:
                next_sec["text"] = sec["text"] + "\n\n" + next_sec["text"] if next_sec["text"] else sec["text"]
            result.pop(i)
        else:
            i += 1

    # Backward pass: merge trailing stub into preceding section (only if preceding is level > 0 and substantial)
    if len(result) >= 2:
        last = result[-1]
        prev = result[-2]
        if (last["level"] > 0
                and _word_count(last["text"]) < STUB_WORDS
                and prev["level"] > 0
                and _word_count(prev["text"]) >= STUB_WORDS):
            if last["text"]:
                prev["text"] = prev["text"] + "\n\n" + last["text"] if prev["text"] else last["text"]
            result.pop()

    return result


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

    page_title = frontmatter.get("title", pagepath) if frontmatter else pagepath
    page_word_count = _word_count(body)

    # Split body into sections, annotate with paths, merge stubs
    sections = _split_into_sections(body)
    sections = _build_header_stack(sections, page_title)
    sections = _merge_stub_sections(sections)

    # Filter out sections with no text
    sections = [s for s in sections if s["text"].strip()]

    if not sections:
        return []

    # Build all chunks across sections
    all_chunks = []  # list of (text_with_prefix, section, section_path)

    for sec in sections:
        section_path = sec["section_path"]
        section = sec["section"]
        prefix = f"[{section_path}] "

        sec_text = sec["text"].strip()

        # For short sections / short total body, keep as single chunk
        if page_word_count < MIN_CHUNK_WORDS or _word_count(sec_text) < MIN_CHUNK_WORDS:
            all_chunks.append((prefix + sec_text, section, section_path))
        else:
            raw_chunks = _chunk_text(sec_text)
            if not raw_chunks:
                continue

            # Apply overlap within section only
            if len(raw_chunks) > 1:
                overlapped = [raw_chunks[0]]
                for i in range(1, len(raw_chunks)):
                    prev_words = raw_chunks[i - 1].split()
                    overlap_words = prev_words[-OVERLAP_WORDS:]
                    overlapped.append(" ".join(overlap_words) + "\n\n" + raw_chunks[i])
                raw_chunks = overlapped

            for chunk_text in raw_chunks:
                all_chunks.append((prefix + chunk_text, section, section_path))

    total_chunks = len(all_chunks)

    return [
        {
            "id": f"{pagepath}::chunk_{i}",
            "text": text,
            "metadata": {
                **meta,
                "chunk_index": i,
                "section": section,
                "section_path": section_path,
                "page_word_count": page_word_count,
                "total_chunks": total_chunks,
            },
        }
        for i, (text, section, section_path) in enumerate(all_chunks)
    ]
