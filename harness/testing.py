"""Run a step's tests and append the result to its test log."""
import datetime as dt
import re
import subprocess
import sys

from . import config as C
from .util import HarnessError, head

TAIL_LINES = 120


def run(root, step):
    if step == "all":
        files = sorted((root / "tests").glob("test_step*_*.py"))
        if not files:
            raise HarnessError("No tests found in tests/.")
        return subprocess.call([sys.executable, "-m", "pytest", "-q", *map(str, files)], cwd=root)

    step = int(step)
    if step not in C.DOC_REPORTS:
        raise HarnessError("Tests are run for Steps 1-7 (or `all`).")
    files = sorted((root / "tests").glob(f"test_step{step}_*.py"))
    if not files:
        raise HarnessError(f"No tests for Step {step}. The tester agent should write tests/test_step{step}_*.py first.")

    proc = subprocess.run([sys.executable, "-m", "pytest", "-v", "-rA", *map(str, files)],
                          cwd=root, capture_output=True, text=True)
    output = (proc.stdout + proc.stderr).rstrip()
    if "No module named pytest" in output:
        raise HarnessError("pytest is not installed. Run: pip install -r requirements.txt")
    summary = next((l.strip("= ").strip() for l in reversed(output.splitlines()) if re.match(r"^=+ .* =+$", l)), "")
    status = "PASSED" if proc.returncode == 0 else "FAILED"

    log = root / C.test_log_path(step)
    if log.exists():
        text = log.read_text(encoding="utf-8")
    else:
        tmpl = root / "templates" / "test_log.md"
        text = (tmpl.read_text(encoding="utf-8") if tmpl.exists() else "# Step {{STEP}} Test Log\n\n## Runs\n")
        text = text.replace("{{STEP}}", str(step)).replace("{{STEP_TITLE}}", C.STEP_TITLES[step])
    runs = len(re.findall(r"^### Run \d+", text, flags=re.M)) + 1
    lines = output.splitlines()
    shown = "\n".join(lines[-TAIL_LINES:])
    if len(lines) > TAIL_LINES:
        shown = f"... ({len(lines) - TAIL_LINES} earlier lines omitted)\n" + shown
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    entry = (f"\n### Run {runs}: {status}\n\n"
             f"- Time: {stamp}\n- Commit: {head(root)}\n- Result: {summary}\n"
             f"- Files: {', '.join(f.name for f in files)}\n\n```\n{shown}\n```\n")
    log.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")
    print(output)
    print(f"\nStep {step} tests {status} ({summary}). Logged as Run {runs} in {C.test_log_path(step)}.")
    if status == "FAILED":
        print("Record each failure and your decision in the 'Failures and decisions' table of the test log.")
    return proc.returncode
