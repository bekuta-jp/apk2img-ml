from __future__ import annotations

from pathlib import Path
from typing import Iterable

from gensim.models.doc2vec import TaggedDocument


def _iter_token_paths(corpus_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in corpus_dir.rglob("*.txt") if path.is_file())


def _load_tokens(path: Path, *, first_line_only: bool) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if first_line_only:
            line = handle.readline().strip()
            return line.split() if line else []

        tokens: list[str] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tokens.extend(line.split())
        return tokens


def load_documents_for_train(corpus_dir: Path, *, first_line_only: bool = True) -> list[TaggedDocument]:
    docs: list[TaggedDocument] = []
    for path in _iter_token_paths(corpus_dir):
        tokens = _load_tokens(path, first_line_only=first_line_only)
        if not tokens:
            continue
        tag = str(path.relative_to(corpus_dir))
        docs.append(TaggedDocument(tokens, [tag]))
    return docs


def load_documents_for_infer(corpus_dir: Path, *, first_line_only: bool = True) -> list[tuple[str, list[str]]]:
    docs: list[tuple[str, list[str]]] = []
    for path in _iter_token_paths(corpus_dir):
        tokens = _load_tokens(path, first_line_only=first_line_only)
        if not tokens:
            continue
        tag = str(path.relative_to(corpus_dir))
        docs.append((tag, tokens))
    return docs
