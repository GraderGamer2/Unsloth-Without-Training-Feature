# SPDX-License-Identifier: AGPL-3.0-only

import base64
import csv
import hashlib
import io
import sys
from pathlib import Path
from zipfile import ZipFile


def strip_optional_assets(wheel: Path) -> None:
    prefixes = ("studio/backend/assets/docs_ui/", "studio/backend/requirements/engines/")
    marker = "studio/backend/personal-vulkan"
    training_entries = {
        "studio/backend/routes/training.py",
        "studio/backend/routes/training_history.py",
        "studio/backend/routes/export.py",
        "studio/backend/routes/openai_fine_tuning.py",
        "studio/backend/studio_mcp/tools/training.py",
        "studio/backend/studio_mcp/tools/export.py",
    }
    temporary = wheel.with_suffix(".new")
    try:
        with ZipFile(wheel) as source, ZipFile(temporary, "w") as target:
            record, = (name for name in source.namelist() if name.endswith(".dist-info/RECORD"))
            rows = []
            for member in source.infolist():
                name = member.filename
                if name in (record, marker) or name in training_entries or name.startswith(prefixes):
                    continue
                data = source.read(member)
                target.writestr(member, data)
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                rows.append((name, f"sha256={digest}", len(data)))
            target.writestr(marker, b"")
            digest = base64.urlsafe_b64encode(hashlib.sha256(b"").digest()).rstrip(b"=").decode()
            rows.append((marker, f"sha256={digest}", 0))
            buffer = io.StringIO(newline = "")
            csv.writer(buffer).writerows([*rows, (record, "", "")])
            target.writestr(source.getinfo(record), buffer.getvalue())
        temporary.replace(wheel)
    finally:
        temporary.unlink(missing_ok = True)


if __name__ == "__main__":
    strip_optional_assets(Path(sys.argv[1]))
