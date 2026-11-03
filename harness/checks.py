"""Submission checks for students (before submitting) and instructors (when grading).

Each check gives PASS, WARN, or FAIL. WARN means "look at this"; it is not proof of a problem.
"""
import collections
import datetime as dt
import json
import re
from pathlib import Path

from . import config as C
from .util import find_table, git, has_commits, is_git_repo, tag_exists

BIG_COMMIT_LINES = 400
ICON = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, area, msg):
        self.items.append((level, area, msg))

    def ok(self, area, msg):
        self.add("PASS", area, msg)

    def warn(self, area, msg):
        self.add("WARN", area, msg)

    def fail(self, area, msg):
        self.add("FAIL", area, msg)

    def counts(self):
        return collections.Counter(l for l, _, _ in self.items)

    def markdown(self, root):
        c = self.counts()
        out = [f"# Submission check: {root.name}", "",
               f"Run {dt.datetime.now().astimezone().isoformat(timespec='seconds')}: "
               f"{c['PASS']} pass, {c['WARN']} warn, {c['FAIL']} fail.", ""]
        area = None
        for level, a, msg in self.items:
            if a != area:
                out += ["", f"## {a}", ""]
                area = a
            out.append(f"- {ICON[level]} {msg}")
        return "\n".join(out) + "\n"


def _read(root, rel):
    p = root / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


# ---------------------------------------------------------------- files

def check_files(root, r):
    area = "Required files"
    required = (C.TOP_LEVEL_FILES + list(C.DOC_REPORTS.values())
                + [C.review_path(s) for s in range(8)] + [C.test_log_path(s) for s in range(1, 8)])
    missing = [f for f in required if not (root / f).exists()]
    if missing:
        r.fail(area, f"Missing {len(missing)} file(s): {', '.join(missing)}")
    else:
        r.ok(area, f"All {len(required)} required Markdown files are present.")

    todo = [f for f in C.TOP_LEVEL_FILES + list(C.DOC_REPORTS.values())
            if (t := _read(root, f)) and C.TODO_RE.search(t)]
    if todo:
        r.fail(area, f"Unfilled [TODO] placeholders remain in: {', '.join(todo)}")
    else:
        r.ok(area, "No [TODO] placeholders left.")

    agents = (_read(root, "AGENTS.md") or "").lower()
    lacking = [k for k in ("input_data", "first column", "label", "year", "how to run") if k not in agents]
    if agents and lacking:
        r.warn(area, f"AGENTS.md does not mention: {', '.join(lacking)}")

    src = [p for p in (root / "src").rglob("*.py") if p.name != "__init__.py"] if (root / "src").exists() else []
    nb = list(root.glob("*.ipynb")) + list(root.glob("notebooks/*.ipynb"))
    if not src and not nb:
        r.fail(area, "No project code found in src/ (or a notebook).")


# ---------------------------------------------------------------- reviews

def check_reviews(root, r):
    area = "Review reports and decision logs"
    rejected_total = 0
    for step in range(8):
        rel = C.review_path(step)
        text = _read(root, rel)
        if text is None:
            continue
        _, comments = find_table(text, "#", section="Comments")
        _, decisions = find_table(text, "#", section="Decision log")
        if not comments:
            r.warn(area, f"{rel}: the reviewer listed no comments.")
            continue
        dec = {row[0]: row for row in decisions if row and row[0]}
        problems = []
        for row in comments:
            num = row[0]
            d = dec.get(num)
            if not d or len(d) < 3:
                problems.append(f"#{num} has no decision")
                continue
            if not re.match(r"(?i)^(accepted|rejected)", d[1]):
                problems.append(f"#{num} decision is not Accepted or Rejected")
            if len(d[2]) < 5:
                problems.append(f"#{num} has no reason")
            if d[1].lower().startswith("rejected"):
                rejected_total += 1
        if problems:
            r.fail(area, f"{rel}: {'; '.join(problems)}.")
        else:
            r.ok(area, f"{rel}: {len(comments)} comment(s), all with a decision and reason.")
    if rejected_total == 0:
        r.warn(area, "No review comment was rejected anywhere. The reflection asks you to discuss one.")


# ---------------------------------------------------------------- tests

def check_tests(root, r):
    area = "Independent testing"
    for step in range(1, 8):
        files = sorted((root / "tests").glob(f"test_step{step}_*.py"))
        rel = C.test_log_path(step)
        text = _read(root, rel)
        if not files:
            r.fail(area, f"Step {step}: no test files (tests/test_step{step}_*.py).")
        if text is None:
            continue
        _, written = find_table(text, "Test", section="Tests written")
        runs = re.findall(r"^### Run \d+: (PASSED|FAILED)", text, flags=re.M)
        _, failures = find_table(text, "Test", section="Failures and decisions")
        msgs = []
        if not written:
            msgs.append("the 'Tests written' table is empty")
        if not runs:
            r.fail(area, f"{rel}: no test runs logged (use `python harness.py test {step}`).")
            continue
        if runs[-1] == "FAILED":
            msgs.append("the latest run FAILED")
        if "FAILED" in runs:
            bad = [f for f in failures if len(f) < 3 or not re.match(r"(?i)^(accepted|rejected)", f[2])]
            if not failures:
                msgs.append("a run failed but no failure decision is recorded")
            elif bad:
                msgs.append(f"{len(bad)} failure row(s) lack an Accepted/Rejected decision")
        if msgs:
            r.warn(area, f"{rel}: {'; '.join(msgs)}.")
        else:
            r.ok(area, f"{rel}: {len(files)} test file(s), {len(runs)} run(s) "
                       f"({runs.count('FAILED')} failed before fixes).")


# ---------------------------------------------------------------- transcripts

def _session_id(path: Path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 200:
                    break
                ev = json.loads(line)
                if ev.get("sessionId"):
                    return ev["sessionId"]
                p = ev.get("payload")
                if ev.get("type") == "session_meta" and isinstance(p, dict) and p.get("id"):
                    return p["id"]
    except (OSError, json.JSONDecodeError):
        return None
    return None


def transcripts(root):
    out = []
    for f in sorted((root / "transcripts").glob("*")) if (root / "transcripts").exists() else []:
        m = C.TRANSCRIPT_RE.match(f.name)
        out.append((f, m))
    return out


def check_transcripts(root, r):
    area = "Role separation and transcripts"
    items = transcripts(root)
    bad_names = [f.name for f, m in items if not m and f.name != ".gitkeep"]
    if bad_names:
        r.warn(area, f"File names not in the NN_YYYY-MM-DD_role_stepN format: {', '.join(bad_names)}")
    good = [(f, m) for f, m in items if m]
    have = collections.defaultdict(set)
    for f, m in good:
        have[int(m["step"])].add(m["role"])
    missing = []
    for step in range(8):
        for role in C.roles_for_step(step):
            if role not in have[step]:
                missing.append(f"step {step} {role}")
    if missing:
        r.fail(area, f"Missing transcripts: {', '.join(missing)}")
    else:
        r.ok(area, "Every step has a transcript for each required role.")

    tiny = [f.name for f, m in good if f.stat().st_size < 400]
    if tiny:
        r.warn(area, f"Very short transcript(s), check they are complete: {', '.join(tiny)}")

    # Fresh context: the reviewer/tester must not contain another role's prompt or share a session id.
    leaks = []
    marker = {role: f"You are the {role.upper()} agent" for role in C.ROLES}
    for f, m in good:
        if f.suffix not in (".md", ".txt", ".jsonl"):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for other in C.ROLES:
            if other != m["role"] and marker[other] in text:
                leaks.append(f"{f.name} contains the {other} prompt")
    if leaks:
        r.fail(area, "Roles not separated: " + "; ".join(leaks))
    else:
        r.ok(area, "No transcript contains another role's prompt.")

    by_id = collections.defaultdict(set)
    for f, m in good:
        if f.suffix == ".jsonl" and (sid := _session_id(f)):
            by_id[sid].add((m["role"], f.name))
    shared = [names for names in by_id.values() if len({role for role, _ in names}) > 1]
    if shared:
        r.fail(area, "The same agent session was used for different roles: "
               + "; ".join(", ".join(sorted(n for _, n in s)) for s in shared))
    elif by_id:
        r.ok(area, f"{len(by_id)} distinct agent session id(s) found; no session is shared between roles.")


# ---------------------------------------------------------------- git

def git_commits(root):
    if not has_commits(root):
        return []
    rng = ["harness-start..HEAD"] if tag_exists(root, "harness-start") else []
    raw = git(root, "log", "--reverse", "--date=short", "--numstat",
              "--format=@@@%h%x1f%ad%x1f%s", *rng)
    commits, cur = [], None
    for line in raw.splitlines():
        if line.startswith("@@@"):
            h, d, s = line[3:].split("\x1f", 2)
            cur = {"hash": h, "date": d, "subject": s, "files": []}
            commits.append(cur)
        elif line.strip() and cur is not None:
            parts = line.split("\t")
            if len(parts) == 3:
                a = int(parts[0]) if parts[0].isdigit() else 0
                cur["files"].append((parts[2], a))
    tag_re = re.compile(r"^\[(%s)\]" % "|".join(C.COMMIT_TAGS))
    for c in commits:
        m = tag_re.match(c["subject"])
        c["role"] = m.group(1) if m else None
    return commits


def _counts_toward_size(path):
    return not (path.startswith("transcripts/") or path.endswith((".jsonl", ".patch"))
                or re.match(r"tests/step\d_test_log\.md$", path))


def check_git(root, r):
    area = "Git history"
    if not is_git_repo(root):
        r.fail(area, "Not a git repository (the .git folder is missing).")
        return []
    commits = git_commits(root)
    if not commits:
        r.fail(area, "No commits after the harness setup.")
        return []
    untagged = [c for c in commits if not c["role"] and not c["subject"].startswith("Merge")]
    if untagged:
        r.fail(area, f"{len(untagged)} commit(s) without a role tag, e.g., "
               + "; ".join(f"{c['hash']} \"{c['subject'][:50]}\"" for c in untagged[:3]))
    else:
        r.ok(area, f"All {len(commits)} commits have a role tag.")
    per_role = collections.Counter(c["role"] for c in commits)
    r.ok(area, "Commits by role: " + ", ".join(f"{k} {per_role.get(k, 0)}" for k in C.COMMIT_TAGS[:4]))
    for role, need in (("builder", 8), ("reviewer", 8), ("tester", 7)):
        if per_role.get(role, 0) < need:
            r.warn(area, f"Only {per_role.get(role, 0)} [{role}] commit(s); expected at least {need} for Steps 0-7.")
    if len(commits) < 25:
        r.warn(area, f"Only {len(commits)} commits in total; the history may have been squashed or done all at once.")
    big = [(c, sum(a for p, a in c["files"] if _counts_toward_size(p))) for c in commits]
    big = [(c, n) for c, n in big if n > BIG_COMMIT_LINES]
    if big:
        r.warn(area, f"Large commit(s) over {BIG_COMMIT_LINES} added lines: "
               + "; ".join(f"{c['hash']} (+{n}) \"{c['subject'][:40]}\"" for c, n in big[:5]))

    # Plan before code: PLAN.md committed at least twice and reviewed before src/ first changes.
    idx = {i: {p for p, _ in c["files"]} for i, c in enumerate(commits)}
    plan = [i for i, fs in idx.items() if "PLAN.md" in fs]
    review0 = [i for i, fs in idx.items() if C.review_path(0) in fs]
    src = [i for i, fs in idx.items() if any(p.startswith("src/") and p.endswith(".py") for p in fs)]
    if src:
        first = src[0]
        if len(plan) < 2 or plan[1] > first:
            r.warn(area, "PLAN.md was not committed twice (original and revised) before the first code commit.")
        elif not review0 or review0[0] > first:
            r.warn(area, "The plan review was not committed before the first code commit.")
        else:
            r.ok(area, "Plan written, reviewed, and revised before any code was committed.")
    return commits


# ---------------------------------------------------------------- consistency

def check_consistency(root, r, commits):
    area = "Consistency between artifacts"
    dates = collections.defaultdict(set)
    for f, m in transcripts(root):
        if m:
            dates[m["role"]].add(dt.date.fromisoformat(m["date"]))
    orphan = collections.Counter()
    for c in commits:
        if c["role"] in C.ROLES:
            d = dt.date.fromisoformat(c["date"])
            near = {d + dt.timedelta(days=k) for k in (-1, 0, 1)}
            if not near & dates[c["role"]]:
                orphan[c["role"]] += 1
    if orphan:
        r.warn(area, "Role-tagged commits with no transcript of that role within a day: "
               + ", ".join(f"{k} {v}" for k, v in orphan.items()))
    elif commits:
        r.ok(area, "Every role-tagged commit has a transcript of the same role within a day.")

    hashes, bad = set(), []
    sources = [(C.review_path(s), "#", "Decision log", 3) for s in range(8)]
    sources += [(C.test_log_path(s), "Test", "Failures and decisions", 3) for s in range(1, 8)]
    for rel, first, section, col in sources:
        _, rows = find_table(_read(root, rel) or "", first, section=section)
        for row in rows:
            if len(row) > col:
                for h in re.findall(r"\b[0-9a-f]{7,40}\b", row[col]):
                    hashes.add((rel, h))
    for rel, h in sorted(hashes):
        if has_commits(root) and git(root, "cat-file", "-t", h, check=False).strip() != "commit":
            bad.append(f"{h} in {rel}")
    if bad:
        r.warn(area, "Commit hashes in decision logs that do not exist in the history: " + ", ".join(bad[:8]))
    elif hashes:
        r.ok(area, f"All {len(hashes)} commit hashes cited in decision logs exist.")

    refl = _read(root, "REFLECTION.md") or ""
    refs = len(re.findall(r"transcripts/\S+", refl)) + len(re.findall(r"\b[0-9a-f]{7,40}\b", refl))
    if refl and refs < 5:
        r.warn(area, f"REFLECTION.md cites only {refs} transcript file(s) or commit(s); each answer should cite one.")
    elif refl:
        r.ok(area, f"REFLECTION.md cites {refs} transcripts or commits.")


def run_all(root: Path) -> Report:
    r = Report()
    check_files(root, r)
    check_reviews(root, r)
    check_tests(root, r)
    check_transcripts(root, r)
    commits = check_git(root, r)
    check_consistency(root, r, commits)
    return r
