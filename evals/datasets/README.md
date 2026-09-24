# Dataset placement

Place licensed or downloaded datasets under their named subdirectories. Do not commit benchmark files.

Fetch and checksum only the required official artifacts (ten files total: seven
dataset artifacts and three pinned scoring scripts):

```bash
make download-benchmarks
make validate-benchmarks
```

`HF_TOKEN` is read from the environment or the ignored repository `.env`. LoCoMo
is not hosted as an official Hugging Face dataset, so its single JSON artifact is
fetched through the GitHub API at the pinned official commit instead.
