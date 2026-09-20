"""Validate a local external benchmark release without running a model."""

import argparse
import json
from pathlib import Path

from evals.adapters.registry import EXTERNAL_DATASETS, external_adapters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=EXTERNAL_DATASETS, required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    adapter = external_adapters()[args.dataset]
    result = adapter.validate(args.path)
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
