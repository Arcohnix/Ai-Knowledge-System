import re
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.ingestion.base import BaseChunker, RawDocument
from app.retrieval.base import RetrievedChunk


class SensibleTextChunker(BaseChunker):
    """
    Intelligent structural chunker that respects semantic document boundaries
    (markdown headers, paragraphs, sentences, and words) rather than blind fixed-width cuts.
    """

    SEPARATORS = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.DEFAULT_CHUNK_OVERLAP
        )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively breaks text using the most specific viable separator."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split text by the selected separator
        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if s:
                good_splits.append(s)

        current_chunk: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)

            if current_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                if current_chunk:
                    merged = separator.join(current_chunk)
                    final_chunks.append(merged)
                    # Handle overlap by keeping trailing pieces from current_chunk
                    overlap_chunk: List[str] = []
                    overlap_len = 0
                    for rev_piece in reversed(current_chunk):
                        if overlap_len + len(rev_piece) + len(separator) <= self.chunk_overlap:
                            overlap_chunk.insert(0, rev_piece)
                            overlap_len += len(rev_piece) + len(separator)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = overlap_len

                # If single piece itself is larger than chunk_size, recurse with finer separators
                if len(piece) > self.chunk_size and new_separators:
                    sub_chunks = self._split_text_recursive(piece, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    current_chunk.append(piece)
                    current_len += len(piece) + (len(separator) if current_chunk else 0)

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return [c.strip() for c in final_chunks if c.strip()]

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """Splits raw text and attaches offset and size metrics."""
        raw_chunks = self._split_text_recursive(text, self.SEPARATORS)

        chunks_with_offsets: List[Dict[str, Any]] = []
        current_search_idx = 0

        for idx, chunk_str in enumerate(raw_chunks):
            # Locate position in original text
            start_pos = text.find(chunk_str, current_search_idx)
            if start_pos == -1:
                start_pos = current_search_idx
            end_pos = start_pos + len(chunk_str)
            current_search_idx = max(start_pos + 1, current_search_idx)

            chunks_with_offsets.append({
                "chunk_index": idx,
                "content": chunk_str,
                "char_count": len(chunk_str),
                "token_estimate": max(1, len(chunk_str) // 4),
                "start_char": start_pos,
                "end_char": end_pos,
            })

        return chunks_with_offsets

    def chunk(self, documents: List[RawDocument]) -> List[RetrievedChunk]:
        """Implements BaseChunker contract."""
        retrieved: List[RetrievedChunk] = []
        for doc in documents:
            splits = self.split_text(doc.content)
            for split in splits:
                retrieved.append(
                    RetrievedChunk(
                        chunk_id="",
                        document_id=doc.source,
                        content=split["content"],
                        metadata={
                            "chunk_index": split["chunk_index"],
                            "char_count": split["char_count"],
                            "token_estimate": split["token_estimate"],
                            "start_char": split["start_char"],
                            "end_char": split["end_char"],
                            **doc.metadata,
                        },
                    )
                )
        return retrieved
