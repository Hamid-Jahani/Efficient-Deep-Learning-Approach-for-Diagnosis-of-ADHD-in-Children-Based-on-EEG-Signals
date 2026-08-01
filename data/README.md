# Dataset placeholder

This repository does not bundle the EEG dataset. To run training or the full
notebook, place the dataset here (or anywhere, and point `--data-dir` /
`ADHD_DATA_DIR` at it) with this layout:

```
data/adhd_dataset/
├── ADHD_part1/
│   ├── *.set
│   └── *.fdt
├── ADHD_part2/
│   ├── *.set
│   └── *.fdt
├── Control_part1/
│   ├── *.set
│   └── *.fdt
└── Control_part2/
    ├── *.set
    └── *.fdt
```

Each `.set`/`.fdt` pair is one child's EEGLAB-format EEG recording. See the
main [README.md](../README.md#data) for how to obtain the dataset and cite
the paper it accompanies.

Nothing under `data/` is tracked by git (see `.gitignore`).
