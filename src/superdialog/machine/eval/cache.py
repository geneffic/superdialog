from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


class ResponseCache:
    """Disk-backed cache for LLM responses, keyed by model + message hash."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)

    def _path(self, model_id: str, key: str) -> Path:
        safe = model_id.replace("/", "_").replace(":", "_")
        return self._dir / safe / f"{key}.json"

    def put_raw(self, model_id: str, key: str, response: str) -> None:
        path = self._path(model_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"response": response}), encoding="utf-8")

    def get_raw(self, model_id: str, key: str) -> str | None:
        path = self._path(model_id, key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("response")

    def invalidate(self, model_id: str | None = None) -> None:
        if model_id is None:
            if self._dir.exists():
                shutil.rmtree(self._dir)
        else:
            safe = model_id.replace("/", "_").replace(":", "_")
            target = self._dir / safe
            if target.exists():
                shutil.rmtree(target)

    @staticmethod
    def hash_messages(messages: list[dict]) -> str:
        serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()