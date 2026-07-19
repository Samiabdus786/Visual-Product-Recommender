"""
run_pipeline.py

Convenience script to run the full pipeline end-to-end:
  1. Build dataset subset
  2. Extract baseline embeddings
  3. Fine-tune backbone (transfer learning) + extract fine-tuned embeddings
  4. Train Siamese network + extract Siamese embeddings
  5. Evaluate & compare all three

Usage:
    python run_pipeline.py --stage all
    python run_pipeline.py --stage subset
    python run_pipeline.py --stage baseline
    python run_pipeline.py --stage finetune
    python run_pipeline.py --stage siamese
    python run_pipeline.py --stage evaluate
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "subset", "baseline", "finetune", "siamese", "evaluate"],
    )
    args = parser.parse_args()

    if args.stage in ("all", "subset"):
        from data.prepare_subset import build_subset
        build_subset()

    if args.stage in ("all", "baseline"):
        from src.feature_extraction import extract_embeddings
        extract_embeddings()

    if args.stage in ("all", "finetune"):
        from src.transfer_learning import run_transfer_learning
        run_transfer_learning()

    if args.stage in ("all", "siamese"):
        from src.train_siamese import train
        train()

    if args.stage in ("all", "evaluate"):
        from src.evaluate import evaluate_all
        evaluate_all()


if __name__ == "__main__":
    main()
