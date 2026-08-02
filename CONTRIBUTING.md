# Contributing

Contributions that improve correctness, reproducibility, Telugu data handling, or documentation are welcome.

## Getting started

1. Create a virtual environment and install the dependencies in `requirements.txt`.
2. Keep generated corpora, tokenizer artifacts, and checkpoints out of commits. They are ignored because they are large and may have separate distribution or licensing requirements.
3. Make a focused change and run the relevant script in `pipeline/` before opening a pull request.

## Pull requests

- Explain the motivation and behavior changed.
- Document configuration and hardware assumptions for model or training changes.
- Add or update a small executable check in `pipeline/` when changing a core component.
- Do not commit secrets, private datasets, model checkpoints, or generated binary files.

## Issues

Include the Python and PyTorch versions, operating system, command used, configuration, and a minimal error trace.
