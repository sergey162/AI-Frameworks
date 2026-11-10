"""Выполнить оба notebook тем Python, которым запущен этот скрипт."""
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

LAB = Path(__file__).resolve().parent
# Все конфигурационные файлы остаются в доступных временных каталогах.
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/vkr_jupyter_runtime")
os.environ.setdefault("IPYTHONDIR", "/tmp/vkr_ipython")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vkr_matplotlib")
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

for path in [LAB / "knn_research.ipynb", LAB.parent / "comparison.ipynb"]:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=3600, resources={"metadata": {"path": str(path.parent)}})
    # Не полагаемся на системный kernel: используем исполняемый файл текущего venv.
    client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client.on_cell_start = lambda cell, cell_index, **kwargs: print(f"{path.name}: cell {cell_index + 1}", flush=True)
    print(f"Выполняется {path} через {sys.executable}", flush=True)
    try:
        client.execute()
    finally:
        # Даже при ошибке сохраняем выполненные ячейки для диагностики.
        nbformat.write(notebook, path)
    print(f"Сохранён {path.name}", flush=True)
