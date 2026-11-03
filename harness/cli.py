"""Command line for the Project 2 harness.  Run `python harness.py --help`."""
import argparse
import importlib.util
import re
import shutil
import sys
from pathlib import Path

from . import config as C
from .util import HarnessError, find_table, git, has_commits, is_git_repo, load_session, tag_exists


# ---------------------------------------------------------------- setup

def cmd_setup(root, args):
    if sys.version_info < (3, 9):
        raise HarnessError("Python 3.9 or newer is required.")
    if not shutil.which("git"):
        raise HarnessError("git is not installed or not on PATH.")
    if not is_git_repo(root):
        git(root, "init")
        print("Created a git repository.")
    git(root, "config", "core.hooksPath", "hooks")
    hook = root / "hooks" / "commit-msg"
    if hook.exists():
        hook.chmod(0o755)
    print("Installed the commit-message hook (role tags are now required).")

    missing = [m for m in ("pandas", "numpy", "sklearn", "scipy", "pytest") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"Missing Python packages: {', '.join(missing)}.  Install with:  pip install -r requirements.txt")
    else:
        print("Python packages: OK.")

    data = list((root / "input_data").glob("*.csv"))
    print(f"input_data/: {len(data)} CSV file(s) found." if data else
          "input_data/: no CSV files yet. Copy the dataset files into input_data/ (they are not committed).")

    if not has_commits(root):
        git(root, "add", "-A")
        git(root, "commit", "-m", "[harness] Start Project 2 from the harness template")
        print("Made the first commit.")
    if not tag_exists(root, "harness-start"):
        git(root, "tag", "harness-start")
        print("Tagged the starting point as harness-start.")
    print("\nSetup done. Next: fill in AGENTS.md, then run  python harness.py session builder 0")


# ---------------------------------------------------------------- status

def _filled(root, rel):
    p = root / rel
    return p.exists() and not C.TODO_RE.search(p.read_text(encoding="utf-8", errors="ignore"))


def _step_status(root, step):
    from .checks import transcripts
    roles = {m["role"] for f, m in transcripts(root) if m and int(m["step"]) == step}
    review = root / C.review_path(step)
    rtext = review.read_text(encoding="utf-8", errors="ignore") if review.exists() else ""
    _, comments = find_table(rtext, "#", section="Comments")
    _, decisions = find_table(rtext, "#", section="Decision log")
    decided = {d[0] for d in decisions if len(d) > 1 and re.match(r"(?i)^(accepted|rejected)", d[1])}
    s = {
        "builder": "builder" in roles,
        "report": _filled(root, C.DOC_REPORTS.get(step, "PLAN.md")),
        "reviewer": "reviewer" in roles,
        "review": bool(comments),
        "decisions": bool(comments) and all(c[0] in decided for c in comments),
    }
    if step > 0:
        log = root / C.test_log_path(step)
        ltext = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
        runs = re.findall(r"^### Run \d+: (PASSED|FAILED)", ltext, flags=re.M)
        s.update({
            "tester": "tester" in roles,
            "tests": bool(list((root / "tests").glob(f"test_step{step}_*.py"))),
            "runs": bool(runs),
            "passing": bool(runs) and runs[-1] == "PASSED",
        })
    return s


NEXT = {
    "builder": "python harness.py session builder {s}",
    "report": "have the builder fill in {doc}",
    "reviewer": "python harness.py session reviewer {s}",
    "review": "the reviewer writes {review}",
    "decisions": "add your Accepted/Rejected decision for every comment in {review}",
    "tester": "python harness.py session tester {s}",
    "tests": "the tester writes tests/test_step{s}_*.py",
    "runs": "python harness.py test {s}",
    "passing": "send failures to the builder (python harness.py session builder {s}), then rerun the tests",
}


def cmd_status(root, args):
    cols = ["builder", "report", "reviewer", "review", "decisions", "tester", "tests", "runs", "passing"]
    print(f"{'Step':<6}" + "".join(f"{c:<11}" for c in cols))
    first_next = None
    for step in range(8):
        s = _step_status(root, step)
        print(f"{step:<6}" + "".join(f"{('yes' if s[c] else '--') if c in s else '':<11}" for c in cols))
        if first_next is None:
            for c in cols:
                if c in s and not s[c]:
                    first_next = NEXT[c].format(s=step, doc=C.DOC_REPORTS.get(step, "PLAN.md"),
                                                review=C.review_path(step))
                    break
    sess = load_session(root)
    if sess:
        print(f"\nIn progress: {sess['role']} session for Step {sess['step']} (save it with `python harness.py save`).")
    print(f"\nNext: {first_next}" if first_next else "\nAll steps complete. Run `python harness.py check`.")


# ---------------------------------------------------------------- main

def main(argv=None):
    root = C.ROOT
    p = argparse.ArgumentParser(prog="harness.py", description="Project 2 agentic AI harness.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="one-time setup: git repo, commit hook, package check, starting tag")

    s = sub.add_parser("session", help="start a fresh builder, reviewer, or tester session")
    s.add_argument("role", choices=C.ROLES)
    s.add_argument("step", type=int, choices=range(8), metavar="STEP")
    s.add_argument("--launch", choices=["claude", "codex"], help="start the agent CLI with the prompt")
    s.add_argument("--force", action="store_true", help="discard an unsaved session and start a new one")

    v = sub.add_parser("save", help="save the transcript of the current session")
    v.add_argument("--file", help="exported chat file (any tool); otherwise Claude Code/Codex logs are found")
    v.add_argument("--tool", choices=["claude", "codex"], help="where to look for the session log")

    t = sub.add_parser("test", help="run a step's tests and log the result")
    t.add_argument("step", help="1-7, or 'all' to run every test without logging")

    sub.add_parser("status", help="show progress through each step and what to do next")

    c = sub.add_parser("check", help="check the submission (students before submitting; instructors when grading)")
    c.add_argument("--repo", help="check another repository (instructors)")
    c.add_argument("--out", help="also write the report to this Markdown file")

    args = p.parse_args(argv)
    try:
        if args.cmd == "setup":
            cmd_setup(root, args)
        elif args.cmd == "session":
            from .sessions import start
            start(root, args.role, args.step, args.launch, args.force)
        elif args.cmd == "save":
            from .transcripts import save
            save(root, args.file, args.tool)
        elif args.cmd == "test":
            from .testing import run
            return run(root, args.step)
        elif args.cmd == "status":
            cmd_status(root, args)
        elif args.cmd == "check":
            from .checks import run_all
            target = Path(args.repo).resolve() if args.repo else root
            report = run_all(target)
            text = report.markdown(target)
            print(text)
            if args.out:
                Path(args.out).write_text(text, encoding="utf-8")
                print(f"Report written to {args.out}")
            return 1 if report.counts()["FAIL"] else 0
    except HarnessError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0
