"""Key file and Claude key rotation.

Key file (default ~/.actspace/api_keys.env; permissions must be 600 and the owner must be the current user): ANTHROPIC_API_KEY_1,
_2, ... (may be empty) and OPENAI_API_KEY. Only the scripts and jobs themselves read it; logs only show numbers such as
"anthropic key #2". File contents are never echoed when parsing fails.

Key pool: at job start all non-empty ANTHROPIC_API_KEY_n are read and used in numeric order, one at a time (the lowest-numbered usable
key). The ledger <results root>/_keys/anthropic_spend.json stores only numbers, amounts and exhaustion reasons, accumulated across
jobs and models. Jobs on several nodes write it concurrently: every read/write holds an fcntl lock and rewrites in place (measured on
NFSv3: 2 nodes x 1,500 increments each with no loss, 5-24 ms per operation; os.replace is not used because other nodes may read the
old file while it is being swapped).
- Proactive switch: once a key has accumulated key_budget_usd (registry, default 190), switch to the next one.
- Quota exhausted: after quota_error recognizes it, the key is marked, the next one is used, and the same call is resent; this failure
  is not billed and the episode is not interrupted. Other 400 / 401 / 403 errors remain fatal.
- All exhausted: KeysExhausted, status = stopped_quota_exhausted, exit code 6; finished episodes are kept. After adding keys, continue
  with --resume. When replacing the key behind an exhausted number, use --reset-key.
Key switches and exhaustion are recorded in _keys/anthropic_events.jsonl (error types and messages with keys removed) and in the job log.

Basis for recognizing quota errors (official documentation, read 2026-11-04):
- https://platform.claude.com/docs/en/api/errors: "402 - billing_error: There's an issue with your billing or payment information.";
  "The API also returns a 400 when usage reaches an organization or workspace spend limit you set"; a 429 may also be
  "reached its usage tier's monthly spend cap".
- https://platform.claude.com/docs/en/api/rate-limits#reaching-your-spend-cap: "On the Messages API, error.details.error_code is enforced_spend_limit_reached.
  Use it to tell this response apart from a rate limit."
- The official pages do not give the exact text of the 400s for insufficient credit balance and for a self-set limit, so these two
  are matched on the invalid_request_error message text (credit balance, usage limit, spend limit).
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import stat
import threading
import time
from pathlib import Path

from ..models import ModelError

DEFAULT_KEY_FILE = Path(os.environ.get("ACTSPACE_KEY_FILE") or Path.home() / ".actspace" / "api_keys.env")   # optional; without it, keys are read from environment variables
DEFAULT_BUDGET_USD = 190.0
QUOTA_STATUS = "stopped_quota_exhausted"
REFRESH_S = 5.0     # ledger writes by other jobs become visible after at most this long; quota errors before that are handled by exhaust
_LINE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)")
_NUMBERED = re.compile(r"ANTHROPIC_API_KEY_(\d+)")
_QUOTA_TEXT = re.compile(r"credit balance|usage limit|spend(?:ing)? limit", re.I)
_LEDGER_LOCK = threading.Lock()
_pool = None


class KeyFileError(Exception):
    pass


class KeyQuota(Exception):
    """This key's quota is exhausted (passed only inside the adapter). info: kind, status, type, message (keys removed)."""

    def __init__(self, info: dict):
        super().__init__(info["kind"])
        self.info = info


class KeysExhausted(ModelError):
    stop_status = QUOTA_STATUS

    def __init__(self, msg: str):
        super().__init__(msg, retryable=False)


def read_key_file(path) -> dict:
    """{variable name: value}, non-empty values only. Refuses on wrong permissions or owner. Tests (including subprocesses they start) never read the real key file."""
    p = Path(path)
    if os.environ.get("PYTEST_CURRENT_TEST") and p.resolve() == DEFAULT_KEY_FILE.resolve():
        raise KeyFileError("tests must not read the real key file")
    try:
        st = p.stat()
    except FileNotFoundError:
        raise KeyFileError(f"{p} does not exist") from None
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise KeyFileError(f"permissions of {p} are {stat.S_IMODE(st.st_mode):o}; must be 600")
    if st.st_uid != os.getuid():
        raise KeyFileError(f"{p} is not owned by the current user")
    out = {}
    for line in p.read_text().splitlines():
        m = _LINE.fullmatch(line)
        if m is None or line.lstrip().startswith("#"):
            continue
        v = m.group(2).strip()
        v = v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"" else v.split("#")[0].strip()
        if v:
            out[m.group(1)] = v
    return out


def anthropic_keys(values: dict) -> dict:
    """{number: key}, sorted by number."""
    found = {int(m.group(1)): v for k, v in values.items() if (m := _NUMBERED.fullmatch(k))}
    return dict(sorted(found.items()))


def ledger_path(root: Path) -> Path:
    return Path(root) / "_keys" / "anthropic_spend.json"


@contextlib.contextmanager
def _ledger(path: Path, write: bool):
    """Reads the ledger under the lock; with write, writes it back in place on exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LEDGER_LOCK, os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o644), "r+") as f:      # the fcntl lock only covers processes; threads rely on _LEDGER_LOCK
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            raw = f.read()
            state = json.loads(raw) if raw.strip() else {"keys": {}}
            yield state
            if write:
                f.seek(0)
                f.write(json.dumps(state, indent=1, sort_keys=True) + "\n")
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def read_ledger(root: Path) -> dict:
    p = ledger_path(root)
    if not p.exists():
        return {"keys": {}}
    with _ledger(p, False) as state:
        return state


def reset_key(root: Path, no: int):
    """The key behind this number was replaced: reset the total and clear the exhausted mark."""
    with _ledger(ledger_path(root), True) as state:
        state["keys"].pop(str(no), None)
    _event(ledger_path(root), {"event": "reset", "key": no})


def _event(ledger: Path, rec: dict):
    rec = {"time": time.strftime("%F %T"), "job": os.environ.get("SLURM_JOB_ID"), **rec}
    with open(ledger.with_name("anthropic_events.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class KeyPool:
    def __init__(self, keys: dict, root: Path, budget: float = DEFAULT_BUDGET_USD, model: str | None = None):
        self.keys, self.path, self.budget, self.model = dict(sorted(keys.items())), ledger_path(root), float(budget), model
        self._lock = threading.Lock()
        self._state, self._read_at, self._no = {"keys": {}}, 0.0, None

    def _sync(self, change=None):
        with _ledger(self.path, change is not None) as state:
            if change is not None:
                change(state)
            self._state, self._read_at = json.loads(json.dumps(state)), time.monotonic()

    def _usable(self, no: int) -> bool:
        r = self._state["keys"].get(str(no)) or {}
        return not r.get("exhausted") and r.get("spent_usd", 0.0) < self.budget

    def _say(self, msg: str, rec: dict):
        print(f"== {msg}", flush=True)
        _event(self.path, {"model": self.model, **rec})

    def current(self) -> tuple[int, str]:
        """(number, key): the lowest-numbered usable key. Raises KeysExhausted when all are exhausted."""
        if time.monotonic() - self._read_at > REFRESH_S:
            self._sync()
        no = next((n for n in self.keys if self._usable(n)), None)
        if no is None:
            self._sync()
            no = next((n for n in self.keys if self._usable(n)), None)
        if no is None:
            msg = f"all anthropic keys are exhausted (numbers {list(self.keys)})"
            self._say(msg, {"event": "all_exhausted", "keys": list(self.keys)})
            raise KeysExhausted(msg)
        with self._lock:
            before, self._no = self._no, no
        if before != no:
            self._say(f"using anthropic key #{no}" + (f" (previously #{before})" if before is not None else ""), {"event": "use", "key": no, "from": before})
        return no, self.keys[no]

    def charge(self, no: int, usd: float):
        def add(state):
            r = state["keys"].setdefault(str(no), {"spent_usd": 0.0, "exhausted": None})
            r["spent_usd"] = round(r["spent_usd"] + usd, 6)
            if r["spent_usd"] >= self.budget and not r["exhausted"]:
                r["exhausted"] = "budget"
        self._sync(add)

    def exhaust(self, no: int, info: dict):
        """The server reported exhausted quota. Recorded once even if several workers report the same key at the same time."""
        first = []

        def mark(state):
            r = state["keys"].setdefault(str(no), {"spent_usd": 0.0, "exhausted": None})
            if not r["exhausted"]:
                r["exhausted"] = info["kind"]
                first.append(True)
        self._sync(mark)
        if first:
            self._say(f"anthropic key #{no} quota exhausted: {info['kind']} (HTTP {info['status']} {info['type']})", {"event": "exhausted", "key": no, **info})


def active_pool() -> KeyPool | None:
    return _pool


def activate(pool: KeyPool | None):
    global _pool
    _pool = pool


def quota_error(status: int, text: str) -> dict | None:
    """Whether an error returned by the server means the account quota is exhausted. See the module docstring for the basis."""
    try:
        err = (json.loads(text) or {}).get("error") or {}
    except (ValueError, AttributeError):
        err = {}
    typ, msg = err.get("type"), str(err.get("message") or text)
    code = (err.get("details") or {}).get("error_code")
    if status == 402 or typ == "billing_error":
        kind = "billing_error"
    elif status == 429 and (code == "enforced_spend_limit_reached" or _QUOTA_TEXT.search(msg)):
        kind = "spend_cap"
    elif status == 400 and typ == "invalid_request_error" and _QUOTA_TEXT.search(msg):
        kind = "credit_balance" if "credit" in msg.lower() else "spend_limit"
    else:
        return None
    return {"kind": kind, "status": status, "type": typ, "message": msg[:500]}


def setup_keys(root: Path, short: str, entry: dict, key_file=None) -> dict:
    """Reads the key file and prepares the keys this process will use. Not needed for mock models without a key file. Returns a note containing only key numbers."""
    if entry["key_env"] is None and key_file is None:
        return {"note": "key: not needed (mock)"}
    if key_file is None and not DEFAULT_KEY_FILE.exists():
        # no key file: read from environment variables (OpenAI uses key_env, e.g. OPENAI_API_KEY; Anthropic rotates ANTHROPIC_API_KEY_1..n or uses a single ANTHROPIC_API_KEY)
        path = "environment"
        values = {k: v for k, v in os.environ.items() if v and (k == entry["key_env"] or k.startswith("ANTHROPIC_API_KEY"))}
    else:
        path = Path(key_file) if key_file else DEFAULT_KEY_FILE
        values = read_key_file(path)
    if entry["provider"] == "anthropic":
        numbered = anthropic_keys(values)
        if not numbered and values.get(entry["key_env"] or "ANTHROPIC_API_KEY"):
            numbered = {1: values[entry["key_env"] or "ANTHROPIC_API_KEY"]}
        if not numbered:
            raise KeyFileError(f"{path} contains no non-empty ANTHROPIC_API_KEY_n")
        activate(KeyPool(numbered, root, entry.get("key_budget_usd", DEFAULT_BUDGET_USD), short))
        return {"note": f"anthropic keys: {' '.join(f'#{n}' for n in numbered)} ({len(numbered)} non-empty)", "keys": list(numbered)}
    var = entry["key_env"]
    if var is None:
        return {"note": "key: not needed (mock)"}
    if not values.get(var):
        raise KeyFileError(f"{var} is empty in {path}")
    os.environ[var] = values[var]       # only within this process, never written to any file
    return {"note": f"{var}: set"}


_SUSPECT = re.compile(r"(?<![A-Za-z0-9])sk-(?:ant-)?[A-Za-z0-9_\-]{8,}")


def scan_for_keys(paths, values=()) -> int:
    """Number of these files that contain something resembling a key: a string starting with sk- or sk-ant-, or the verbatim text of any of values. Does not return what matched."""
    hits = 0
    needles = [v.encode() for v in values]
    for p in paths:
        data = Path(p).read_bytes()
        hits += bool(_SUSPECT.search(data.decode("utf-8", "replace"))) or any(n in data for n in needles)
    return hits
