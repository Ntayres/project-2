"""Save an agent session transcript into transcripts/ under the right name."""
import datetime as dt
import json
import re
import shutil
from pathlib import Path

from .util import HarnessError, load_session, save_session, state_dir

MAX_OUTPUT = 3000  # characters of each tool result kept in the readable .md copy


def _claude_dir(root: Path) -> Path:
    encoded = re.sub(r"[^A-Za-z0-9]", "-", str(root.resolve()))
    return Path.home() / ".claude" / "projects" / encoded


def _newest(files, since: float):
    files = [f for f in files if f.stat().st_mtime >= since]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def find_session_file(root: Path, started: str, tool: str = None):
    since = dt.datetime.fromisoformat(started).timestamp() - 60
    if tool in (None, "claude"):
        d = _claude_dir(root)
        if d.exists():
            f = _newest(list(d.glob("*.jsonl")), since)
            if f:
                return f, "claude"
    if tool in (None, "codex"):
        d = Path.home() / ".codex" / "sessions"
        if d.exists():
            cands = [f for f in d.rglob("rollout-*.jsonl") if f.stat().st_mtime >= since]
            cands = [f for f in cands if str(root.resolve()) in f.read_text(encoding="utf-8", errors="ignore")[:20000]]
            f = _newest(cands, since)
            if f:
                return f, "codex"
    return None, None


def _blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def _short(s, n=MAX_OUTPUT):
    s = s if isinstance(s, str) else json.dumps(s, indent=1, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n} more characters in the .jsonl file]"


def render_jsonl(lines, title: str) -> str:
    """Readable markdown from a Claude Code or Codex JSONL session log."""
    out = [f"# {title}\n"]
    for raw in lines:
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") or ev.get("payload") or {}
        if not isinstance(msg, dict):
            continue
        ts = ev.get("timestamp", "")
        ptype = msg.get("type")
        if ptype == "function_call":  # Codex tool call
            out.append(f"**Tool call: {msg.get('name', '')}** {ts}\n\n```\n{_short(msg.get('arguments', ''))}\n```\n")
            continue
        if ptype == "function_call_output":
            out.append(f"**Tool result** {ts}\n\n```\n{_short(msg.get('output', ''))}\n```\n")
            continue
        role = msg.get("role") or ev.get("type")
        if role not in ("user", "assistant"):
            continue
        for b in _blocks(msg.get("content")):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t in ("text", "input_text", "output_text") and b.get("text", "").strip():
                out.append(f"## {role.capitalize()} {ts}\n\n{b['text'].strip()}\n")
            elif t == "tool_use":
                out.append(f"**Tool call: {b.get('name', '')}** {ts}\n\n```\n{_short(b.get('input', ''))}\n```\n")
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                out.append(f"**Tool result** {ts}\n\n```\n{_short(c or '')}\n```\n")
    return "\n".join(out)


def save(root: Path, file: str = None, tool: str = None):
    sess = load_session(root)
    if not sess:
        raise HarnessError("No session is in progress. Start one with `python harness.py session ROLE STEP`.")
    tdir = root / "transcripts"
    tdir.mkdir(exist_ok=True)
    name = sess["filename"]
    title = f"Transcript: {sess['role']}, Step {sess['step']} (started {sess['started']})"

    if file:
        src = Path(file).expanduser()
        if not src.exists():
            raise HarnessError(f"File not found: {src}")
        if src.suffix == ".jsonl":
            shutil.copyfile(src, tdir / f"{name}.jsonl")
            md = render_jsonl(src.read_text(encoding="utf-8", errors="replace").splitlines(), title)
            (tdir / f"{name}.md").write_text(md, encoding="utf-8")
        else:
            ext = ".md" if src.suffix.lower() in (".md", ".markdown") else ".txt"
            shutil.copyfile(src, tdir / f"{name}{ext}")
        source = str(src)
    else:
        src, found = find_session_file(root, sess["started"], tool or sess.get("tool"))
        if not src:
            raise HarnessError(
                "Could not find the session automatically. Export the chat from your tool "
                "(Claude Code: /export; Cursor or Copilot: export or copy the chat to a file), then run "
                "`python harness.py save --file PATH`.")
        shutil.copyfile(src, tdir / f"{name}.jsonl")
        md = render_jsonl(src.read_text(encoding="utf-8", errors="replace").splitlines(), title)
        (tdir / f"{name}.md").write_text(md, encoding="utf-8")
        source = f"{found} session log {src.name}"

    with open(state_dir(root) / "sessions.log", "a", encoding="utf-8") as log:
        log.write(json.dumps({**sess, "saved": dt.datetime.now().astimezone().isoformat(timespec='seconds'),
                              "source": source}) + "\n")
    save_session(root, None)
    print(f"Saved transcript to transcripts/{name}.* (from {source}).")
    print(f"Commit it, e.g.:  git add transcripts && git commit -m \"[{sess['role']}] Step {sess['step']} transcript\"")
