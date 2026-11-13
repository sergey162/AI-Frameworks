"""Выполнить исследования и общее сравнение текущим интерпретатором Python."""

import argparse
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient


def execute(path):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=3600, resources={"metadata": {"path": str(path.parent)}})
    client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client.on_cell_start = lambda cell, cell_index, **kwargs: print(f"{path.name}: cell {cell_index + 1}", flush=True)
    print(f"Выполняется {path.name} через {sys.executable}", flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print(f"Сохранён {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", nargs="+", choices=[1, 2], type=int, default=[1, 2])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name, value in {"JUPYTER_RUNTIME_DIR": "/tmp/vkr_jupyter_runtime",
                        "IPYTHONDIR": "/tmp/vkr_ipython", "MPLCONFIGDIR": "/tmp/vkr_matplotlib"}.items():
        os.environ.setdefault(name, value)
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[name] = "1"
    paths = {1: root / "lab_1" / "knn_research.ipynb", 2: root / "lab_2" / "linear_research.ipynb"}
    for number in sorted(set(args.labs)):
        execute(paths[number])
    execute(root / "comparison.ipynb")


if __name__ == "__main__":
    main()
