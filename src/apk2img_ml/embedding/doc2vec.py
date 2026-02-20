from __future__ import annotations

from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from typing import Sequence

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_doc2vec(
    documents: Sequence[TaggedDocument],
    *,
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 20,
) -> Doc2Vec:
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )
    model.build_vocab(documents)

    total_examples = len(documents)
    start_alpha = model.alpha
    end_alpha = model.min_alpha
    alpha_delta = (start_alpha - end_alpha) / epochs if epochs > 0 else 0.0

    for _ in range(epochs):
        model.train(documents, total_examples=total_examples, epochs=1)
        model.alpha = max(model.alpha - alpha_delta, end_alpha)
        model.min_alpha = model.alpha

    return model


def infer_vectors(
    model: Doc2Vec,
    docs: Sequence[tuple[str, list[str]]],
    *,
    infer_epochs: int = 50,
    alpha: float = 0.025,
    workers: int = 1,
) -> list[tuple[str, list[float]]]:
    def infer_one(task: tuple[int, str, list[str]]) -> tuple[int, str, list[float]]:
        idx, tag, tokens = task
        vec = model.infer_vector(tokens, epochs=infer_epochs, alpha=alpha)
        return idx, tag, vec.tolist()

    tasks = [(idx, tag, tokens) for idx, (tag, tokens) in enumerate(docs)]

    if workers <= 1:
        items = [infer_one(task) for task in tasks]
    else:
        with ThreadPool(processes=workers) as pool:
            items = list(pool.imap_unordered(infer_one, tasks))

    items.sort(key=lambda item: item[0])
    return [(tag, vec) for _, tag, vec in items]


def save_vectors_tsv(vectors: Sequence[tuple[str, Sequence[float]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if vectors:
        dim = len(vectors[0][1])
    else:
        dim = 0

    with out_path.open("w", encoding="utf-8") as handle:
        header = ["tag"] + [f"v{i}" for i in range(dim)]
        handle.write("\t".join(header) + "\n")

        for tag, vector in vectors:
            handle.write(f"{tag}\t" + "\t".join(str(x) for x in vector) + "\n")
