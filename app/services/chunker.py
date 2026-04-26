"""텍스트 청킹.

RAG 인덱싱 시 긴 텍스트를 청크로 분할. 봇 설정의 splitter/size/overlap 을 반영.
프론트의 chunk-preview.ts 와 의도적으로 동일한 알고리즘을 사용해 미리보기와 결과를 일치시킨다.

splitter 종류:
  recursive : 빈 줄 → 줄바꿈 → 마침표 → 공백 순 fallback (기본, 가장 보편적)
  sentence  : 문장 단위 (.!?)
  paragraph : 빈 줄(\\n\\n) 단위
  fixed     : N자 고정 분할
"""

import re
from typing import List, Literal

ChunkSplitter = Literal["recursive", "sentence", "paragraph", "fixed"]


def chunk_text(
    content: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    splitter: ChunkSplitter = "recursive",
) -> List[str]:
    """텍스트를 청크 리스트로 분할. 빈 입력은 [] 반환."""
    text = (content or "").strip()
    if not text:
        return []

    # 안전장치 — overlap >= size 면 무한루프 위험. 그냥 통째로 1청크.
    if chunk_overlap >= chunk_size:
        return [text]

    splitter = (splitter or "recursive").lower()  # type: ignore[assignment]

    if splitter == "fixed":
        return _split_fixed(text, chunk_size, chunk_overlap)
    if splitter == "paragraph":
        parts = _split_by_regex(text, r"\n\s*\n")
        return _group_parts(parts, chunk_size, chunk_overlap, "\n\n")
    if splitter == "sentence":
        parts = _split_by_regex(text, r"(?<=[.!?])\s+")
        return _group_parts(parts, chunk_size, chunk_overlap, " ")
    # recursive 가 기본 + 알 수 없는 값이면 fallback
    return _split_recursive(text, chunk_size, chunk_overlap)


# =====================================================================
# 분할 알고리즘
# =====================================================================


def _split_fixed(text: str, size: int, overlap: int) -> List[str]:
    """N자 고정 분할 (오버랩 적용)."""
    stride = max(1, size - overlap)
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        if i + size >= len(text):
            break
        i += stride
    return chunks


def _split_by_regex(text: str, pattern: str) -> List[str]:
    """정규식으로 split + 빈 부분 제거."""
    return [p.strip() for p in re.split(pattern, text) if p and p.strip()]


def _split_recursive(text: str, size: int, overlap: int) -> List[str]:
    """우선순위 구분자로 fallback 분할.

    - 빈 줄 → 줄바꿈 → 마침표 → 공백 순.
    - 어떤 단계에서든 모든 part 가 size 이내면 그걸로 group.
    - 큰 part 가 남으면 fixed 로 추가 분할 후 group.
    """
    separators = [
        (r"\n\s*\n", "\n\n"),
        (r"\n", "\n"),
        (r"(?<=[.!?])\s+", " "),
        (r"\s+", " "),
    ]
    for pattern, joiner in separators:
        parts = _split_by_regex(text, pattern)
        if len(parts) <= 1:
            continue

        if all(len(p) <= size for p in parts):
            return _group_parts(parts, size, overlap, joiner)

        # 큰 part 가 섞여있으면 fixed 로 평탄화 후 group
        flattened: List[str] = []
        for p in parts:
            if len(p) <= size:
                flattened.append(p)
            else:
                flattened.extend(_split_fixed(p, size, 0))
        return _group_parts(flattened, size, overlap, joiner)

    # 분리 가능한 구분자가 없음 → 그냥 fixed
    return _split_fixed(text, size, overlap)


def _group_parts(parts: List[str], size: int, overlap: int, sep: str) -> List[str]:
    """부분(parts)들을 size까지 채워 청크로 그룹핑. 경계에서 overlap 적용."""
    result: List[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) > size and current:
            result.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}{sep}{part}" if tail else part
        else:
            current = candidate

    if current:
        result.append(current)
    return result
