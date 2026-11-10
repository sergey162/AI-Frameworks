"""Проверить или восстановить оба исходных файла с контролем SHA256."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
from src.experiment import DATASETS

LAB = Path(__file__).resolve().parent
SOURCES = {
    "classification": (
        "https://raw.githubusercontent.com/dpidaparthi/loan-approval-classification/"
        "379a24f61cd5d9f032415b14a5797f30cf1d4a05/data/loan_data.csv"
    ),
    "regression": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
}

for task, spec in DATASETS.items():
    path = LAB / "data" / "raw" / spec["file"]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]:
        print(f"{path.name}: локальный файл соответствует SHA256")
        continue
    with urlopen(SOURCES[task], timeout=90) as response:
        payload = response.read()
    if task == "regression":
        with ZipFile(BytesIO(payload)) as archive:
            payload = archive.read("Concrete_Data.xls")
    if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise ValueError(f"SHA256 {path.name} не совпал; локальный файл не изменён")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    print(f"Сохранён {path}")
