from __future__ import annotations

import argparse
from pathlib import Path

from .extraction import ExtractionBackend, batch_extract
from .extraction.config import TokenFilter
from .extraction.pipeline import BackendOptions


def _add_extract_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("extract", help="Extract API tokens from APK files")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=[x.value for x in ExtractionBackend], default="hybrid")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--failures-csv", type=Path)

    parser.add_argument("--include-prefix", nargs="*", default=["Landroid/", "Ljava/", "Lkotlin/"])
    parser.add_argument("--include-none", action="store_true")
    parser.add_argument("--exclude-prefix", nargs="*", default=["Landroidx/test/"])
    parser.add_argument("--include-signature", action="store_true")
    parser.add_argument("--no-fallback-on-empty", action="store_true")

    parser.add_argument("--baksmali-jar", type=Path, default=Path("baksmali.jar"))
    parser.add_argument("--java-cmd", type=str, default="java")
    parser.add_argument("--baksmali-include-assets", action="store_true")

    parser.add_argument("--apktool-cmd", type=str, default="apktool")
    parser.add_argument("--no-res", action="store_true")
    parser.add_argument("--no-apktool-retry-strip-assets", action="store_true")

    parser.set_defaults(handler=_handle_extract)


def _add_train_doc2vec_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train-doc2vec", help="Train Doc2Vec from token corpus")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("doc2vec.model"))
    parser.add_argument("--docvecs-out", type=Path, default=Path("docvecs.tsv"))
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--read-all-lines", action="store_true")
    parser.set_defaults(handler=_handle_train_doc2vec)


def _add_infer_doc2vec_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("infer-doc2vec", help="Infer vectors from a trained Doc2Vec model")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--docvecs-out", type=Path, default=Path("docvecs_test.tsv"))
    parser.add_argument("--infer-epochs", type=int, default=50)
    parser.add_argument("--infer-alpha", type=float, default=0.025)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--read-all-lines", action="store_true")
    parser.set_defaults(handler=_handle_infer_doc2vec)


def _add_docvec_to_png_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("docvec-to-png", help="Convert Doc2Vec TSV to grayscale PNG images")
    parser.add_argument("--docvecs-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["flat", "grid", "raw32"], default="flat")
    parser.add_argument("--pixels-per-vector", type=int, default=0)
    parser.add_argument("--grid-rows", type=int, default=0)
    parser.add_argument("--grid-cols", type=int, default=0)
    parser.add_argument("--vec-pix-rows", type=int, default=1)
    parser.add_argument("--vec-pix-cols", type=int, default=1)
    parser.set_defaults(handler=_handle_docvec_to_png)


def _handle_extract(args: argparse.Namespace) -> int:
    include_prefixes = None if args.include_none else tuple(args.include_prefix) if args.include_prefix else None
    exclude_prefixes = tuple(args.exclude_prefix) if args.exclude_prefix else None

    token_filter = TokenFilter(
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        include_signature=args.include_signature,
        fallback_on_empty=not args.no_fallback_on_empty,
    )

    options = BackendOptions(
        baksmali_jar=args.baksmali_jar,
        java_cmd=args.java_cmd,
        include_assets_dex=args.baksmali_include_assets,
        apktool_cmd=args.apktool_cmd,
        no_res=args.no_res,
        retry_strip_assets=not args.no_apktool_retry_strip_assets,
    )

    results = batch_extract(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        backend=ExtractionBackend(args.backend),
        token_filter=token_filter,
        options=options,
        workers=max(1, args.workers),
        overwrite=args.overwrite,
        failures_csv=args.failures_csv,
    )

    if not results:
        print("No APK files found.")
        return 0

    ok = sum(1 for item in results if item.ok and not item.skipped)
    skipped = sum(1 for item in results if item.skipped)
    failed = sum(1 for item in results if not item.ok)

    print(f"Completed: ok={ok}, skipped={skipped}, failed={failed}, total={len(results)}")
    if failed:
        print("Failed files:")
        for item in results:
            if not item.ok:
                print(f"- {item.apk_path}: {item.error}")
        return 1

    return 0


def _handle_train_doc2vec(args: argparse.Namespace) -> int:
    from .embedding import load_documents_for_train, save_vectors_tsv, train_doc2vec

    docs = load_documents_for_train(args.corpus_dir, first_line_only=not args.read_all_lines)
    if not docs:
        raise RuntimeError(f"no valid documents found under: {args.corpus_dir}")

    model = train_doc2vec(
        docs,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_out))

    vectors = [(doc.tags[0], model.dv[doc.tags[0]].tolist()) for doc in docs]
    save_vectors_tsv(vectors, args.docvecs_out)

    print(f"Saved model: {args.model_out}")
    print(f"Saved vectors: {args.docvecs_out}")
    return 0


def _handle_infer_doc2vec(args: argparse.Namespace) -> int:
    from gensim.models.doc2vec import Doc2Vec

    from .embedding import infer_vectors, load_documents_for_infer, save_vectors_tsv

    docs = load_documents_for_infer(args.corpus_dir, first_line_only=not args.read_all_lines)
    if not docs:
        raise RuntimeError(f"no valid documents found under: {args.corpus_dir}")

    model = Doc2Vec.load(str(args.model), mmap="r")
    vectors = infer_vectors(
        model,
        docs,
        infer_epochs=args.infer_epochs,
        alpha=args.infer_alpha,
        workers=max(1, args.workers),
    )
    save_vectors_tsv(vectors, args.docvecs_out)

    print(f"Saved inferred vectors: {args.docvecs_out}")
    return 0


def _handle_docvec_to_png(args: argparse.Namespace) -> int:
    from .imaging import load_docvecs_tsv, save_docvecs_as_png

    docs = load_docvecs_tsv(args.docvecs_tsv)
    if not docs:
        raise RuntimeError(f"no vectors found in: {args.docvecs_tsv}")

    save_docvecs_as_png(
        docs,
        args.output_dir,
        mode=args.mode,
        pixels_per_vector=args.pixels_per_vector,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        vec_pix_rows=args.vec_pix_rows,
        vec_pix_cols=args.vec_pix_cols,
    )

    print(f"Saved PNG directory: {args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apk2img-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_extract_parser(subparsers)
    _add_train_doc2vec_parser(subparsers)
    _add_infer_doc2vec_parser(subparsers)
    _add_docvec_to_png_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
