from __future__ import annotations

from io import BytesIO

import pandas as pd


def parity_to_xlsx(matrix_rows: list[dict]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        pd.DataFrame(matrix_rows).to_excel(writer, index=False, sheet_name="Matrix")
    bio.seek(0)
    return bio.read()
