from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class CsvLogger:
    path: str
    fieldnames: list[str]

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fp = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=self.fieldnames)
        if self._fp.tell() == 0:
            self._writer.writeheader()
            self._fp.flush()

    def log(self, row: dict[str, Any]) -> None:
        # Only keep known columns to keep CSV stable
        out = {k: row.get(k, None) for k in self.fieldnames}
        self._writer.writerow(out)
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


