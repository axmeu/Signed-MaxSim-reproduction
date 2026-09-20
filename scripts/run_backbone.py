import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signed_maxsim.config import (
    BACKBONES,
    DEFAULT_N_GENERAL_SAMPLES,
    DEFAULT_NEVIR_OVERSAMPLE,
    set_seed,
)
from signed_maxsim.data import (
    build_train_dataset,
    load_general_train_pool,
    load_nevir_splits,
)
from signed_maxsim.runner import run_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=list(BACKBONES.keys()))
    parser.add_argument("--n-general-samples", type=int, default=DEFAULT_N_GENERAL_SAMPLES)
    parser.add_argument("--nevir-oversample", type=int, default=DEFAULT_NEVIR_OVERSAMPLE)
    args = parser.parse_args()

    set_seed()

    nevir_triplets_dataset, nevir_dev, nevir_test = load_nevir_splits()
    general_train_full = load_general_train_pool()
    train_dataset = build_train_dataset(
        nevir_triplets_dataset, general_train_full,
        n_general_samples=args.n_general_samples,
        nevir_oversample=args.nevir_oversample,
    )
    print(f"Train set : {len(train_dataset)} exemples "
          f"(n_general_samples={args.n_general_samples}, nevir_oversample={args.nevir_oversample})")

    run_backbone(args.backbone, train_dataset, nevir_dev, nevir_test)


if __name__ == "__main__":
    main()
