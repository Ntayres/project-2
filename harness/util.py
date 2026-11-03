"""Small helpers: git calls, session state, markdown tables."""
import json
import subprocess
from pathlib import Path


class HarnessError(Exception):
    pass


def git(root: Path, *args, check=True) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except FileNotFoundError:
        raise HarnessError("git is not installed or not on PATH.")
    if check and out.returncode != 0:
        raise HarnessError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def has_commits(root: Path) -> bool:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True).returncode == 0


def tag_exists(root: Path, tag: str) -> bool:
    return subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                          cwd=root, capture_output=True).returncode == 0


def head(root: Path, short=True) -> str:
    if not has_commits(root):
        return "(no commits)"
    return git(root, "rev-parse", "--short" if short else "--verify", "HEAD").strip()


def state_dir(root: Path) -> Path:
    d = root / ".harness"
    d.mkdir(exist_ok=True)
    return d


def load_session(root: Path):
    f = root / ".harness" / "current_session.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def save_session(root: Path, data):
    f = state_dir(root) / "current_session.json"
    if data is None:
        f.unlink(missing_ok=True)
    else:
        f.write_text(json.dumps(data, indent=2), encoding="utf-8")


def md_tables(text: str):
    """Return every markdown pipe table in text as a list of rows (header row excluded)."""
    tables, cur = [], []
    for line in text.splitlines() + [""]:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cur.append([c.strip() for c in s.strip("|").split("|")])
        else:
            if len(cur) >= 2:
                header, rows = cur[0], [r for r in cur[2:] if any(c for c in r)]
                tables.append((header, rows))
            cur = []
    return tables


def find_table(text: str, first_header: str, section: str = None):
    """Rows of the first table whose first header cell matches, optionally inside a '## section'."""
    if section:
        parts = text.split("\n## ")
        text = next((p for p in parts if p.lower().startswith(section.lower())), "")
    for header, rows in md_tables(text):
        if header and header[0].lower() == first_header.lower():
            return header, rows
    return None, []
