"""Progress snapshots tolerate brief Windows file-sharing conflicts."""
import json
import time


def read_snapshot(path, attempts=4):
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if attempt + 1 < attempts:
                time.sleep(0.025)
    return None


def atomic_write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.05)
