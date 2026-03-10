"""Unit tests for chunking — no ChromaDB required."""

from otterwiki_semantic_search.chunking import chunk_page, OVERLAP_WORDS


class TestChunkPage:
    def test_empty_content(self):
        result = chunk_page("test", "")
        assert result == []

    def test_short_content_single_chunk(self):
        content = "This is a short page with just a few words."
        result = chunk_page("test/page", content)
        assert len(result) == 1
        assert result[0]["id"] == "test/page::chunk_0"
        assert result[0]["text"] == content
        assert result[0]["metadata"]["page_path"] == "test/page"
        assert result[0]["metadata"]["chunk_index"] == 0

    def test_frontmatter_stripped(self):
        content = "---\ntitle: My Page\ncategory: notes\ntags:\n  - foo\n  - bar\n---\nBody text here."
        result = chunk_page("test", content)
        assert len(result) == 1
        assert result[0]["text"] == "Body text here."
        assert result[0]["metadata"]["title"] == "My Page"
        assert result[0]["metadata"]["category"] == "notes"
        assert result[0]["metadata"]["tags"] == "foo, bar"

    def test_frontmatter_crlf(self):
        content = "---\r\ntitle: My Page\r\n---\r\nBody text here."
        result = chunk_page("test", content)
        assert len(result) == 1
        assert "Body text here." in result[0]["text"]
        assert result[0]["metadata"]["title"] == "My Page"

    def test_long_content_multiple_chunks(self):
        # Generate content with enough words to exceed MIN_CHUNK_WORDS (300)
        paragraphs = []
        for i in range(20):
            paragraphs.append(f"Paragraph {i}. " + " ".join(f"word{j}" for j in range(25)))
        content = "\n\n".join(paragraphs)

        result = chunk_page("long/page", content)
        assert len(result) > 1

        # All chunks should have correct metadata
        for i, chunk in enumerate(result):
            assert chunk["id"] == f"long/page::chunk_{i}"
            assert chunk["metadata"]["page_path"] == "long/page"
            assert chunk["metadata"]["chunk_index"] == i

    def test_overlap_between_chunks(self):
        # Generate content that will produce multiple chunks
        paragraphs = []
        for i in range(20):
            paragraphs.append(f"Paragraph {i}. " + " ".join(f"word{j}" for j in range(25)))
        content = "\n\n".join(paragraphs)

        result = chunk_page("test", content)
        assert len(result) > 1, "Expected multiple chunks for overlap test"

        # Second chunk should start with overlap words from end of first chunk
        first_words = result[0]["text"].split()
        tail_of_first = first_words[-10:]
        second_text = result[1]["text"]
        overlap_count = sum(1 for w in tail_of_first if w in second_text)
        assert overlap_count > 0, "Expected overlap between adjacent chunks"

    def test_overlap_is_capped(self):
        """Overlap should never exceed OVERLAP_WORDS, even for short chunks."""
        # Create content with many short paragraphs to produce short chunks
        paragraphs = []
        for i in range(30):
            paragraphs.append(f"Short paragraph number {i} with a few words.")
        content = "\n\n".join(paragraphs)

        result = chunk_page("test", content)
        if len(result) > 1:
            for i in range(1, len(result)):
                # The overlap prefix should be at most OVERLAP_WORDS
                # (it appears before the first \n\n in the chunk)
                parts = result[i]["text"].split("\n\n", 1)
                if len(parts) == 2:
                    overlap_word_count = len(parts[0].split())
                    assert overlap_word_count <= OVERLAP_WORDS

    def test_frontmatter_only_no_body(self):
        content = "---\ntitle: Empty\n---\n"
        result = chunk_page("test", content)
        assert result == []

    def test_chunk_ids_are_unique(self):
        paragraphs = []
        for i in range(20):
            paragraphs.append(" ".join(f"word{j}" for j in range(30)))
        content = "\n\n".join(paragraphs)

        result = chunk_page("test", content)
        ids = [c["id"] for c in result]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_single_large_paragraph_split_on_sentences(self):
        # One huge paragraph with sentence boundaries
        sentences = [f"Sentence number {i} has some content here." for i in range(50)]
        content = " ".join(sentences)

        result = chunk_page("test", content)
        assert len(result) > 1

    def test_tags_as_string(self):
        content = "---\ntags: single-tag\n---\nSome body."
        result = chunk_page("test", content)
        assert result[0]["metadata"]["tags"] == "single-tag"

    def test_whitespace_only_content(self):
        result = chunk_page("test", "   \n\n   \n")
        assert result == []

    def test_sentence_split_on_question_and_exclamation(self):
        # Needs enough words to exceed MIN_CHUNK_WORDS and trigger sentence splitting
        sentences = [
            f"Is this really sentence number {i} with enough extra words to matter?"
            if i % 2 == 0
            else f"Yes it certainly is number {i} and it has plenty of words too!"
            for i in range(60)
        ]
        content = " ".join(sentences)

        result = chunk_page("test", content)
        assert len(result) > 1
