"""Paragraph-aware splitting with a hard character bound."""
import re
from app.config import settings

CHUNKER_VERSION = 'bounded-paragraph-v2'


def chunk_text(text: str, chunk_size: int | None = None) -> list[str]:
    """Split text into nonempty chunks without exceeding a character limit.

    Keep paragraphs together where possible. Split oversized paragraphs near
    whitespace, falling back to a hard character boundary for long unbroken
    strings. Normalize line endings and trim boundary whitespace. This function
    does not count tokens; the embedding layer applies its separate input guard.

    Args:
        text: Extracted document text.
        chunk_size: Maximum characters per chunk; defaults to CHUNK_SIZE.

    Returns:
        Ordered text chunks, or an empty list for whitespace-only input.

    Raises:
        ValueError: The requested character limit is zero or negative.
    """
    size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
    if size <= 0:
        raise ValueError('chunk_size must be positive')
    chunks, current = [], ''
    for paragraph in re.split(r'\n\s*\n', text.replace('\r\n', '\n').replace('\r', '\n')):
        remaining = paragraph.strip()
        while remaining:
            if len(remaining) > size:
                if current:
                    chunks.append(current)
                    current = ''
                boundary = max(remaining.rfind(' ', 0, size + 1), remaining.rfind('\n', 0, size + 1))
                boundary = boundary if boundary >= size // 2 and boundary > 0 else size
                chunks.append(remaining[:boundary].strip())
                remaining = remaining[boundary:].strip()
            else:
                candidate = f'{current}\n\n{remaining}' if current else remaining
                if len(candidate) <= size:
                    current = candidate
                else:
                    chunks.append(current)
                    current = remaining
                break
    if current:
        chunks.append(current)
    return chunks
