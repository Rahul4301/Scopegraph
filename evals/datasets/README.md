# Dataset placement

Place licensed or downloaded datasets under their named subdirectories. Do not commit benchmark files.

Phase 8 adapters accept local releases and validate them before replay:

```bash
make validate-external DATASET=longmemeval DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json
make validate-external DATASET=locomo DATA_PATH=data/locomo/locomo10.json
make validate-external DATASET=memconflict DATA_PATH=data/memconflict/release.json
```

Acquire LongMemEval from its [official repository](https://github.com/xiaowu0162/longmemeval), LoCoMo from the [official repository](https://github.com/snap-research/locomo), and MemConflict from the [official repository](https://github.com/TaoZhen1110/MemConflict). The MemConflict adapter uses a documented local interchange shape because releases vary; it never fabricates records.
