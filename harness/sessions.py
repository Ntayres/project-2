"""Start a builder, reviewer, or tester session with a fresh-context prompt."""
import datetime as dt
import re
import shutil
import subprocess

from . import config as C
from .util import HarnessError, git, has_commits, head, load_session, save_session, state_dir, tag_exists


def _next_number(root):
    nums = [int(m.group("num")) for f in (root / "transcripts").glob("*")
            if (m := C.TRANSCRIPT_RE.match(f.name))]
    return max(nums, default=0) + 1


def _render(template: str, step: int, values: dict) -> str:
    keep, drop = ("step0", "steps") if step == 0 else ("steps", "step0")
    template = re.sub(rf"<!-- {drop} -->.*?<!-- /{drop} -->\n?", "", template, flags=re.S)
    template = re.sub(rf"<!-- /?{keep} -->\n?", "", template)
    for k, v in values.items():
        template = template.replace("{{" + k + "}}", v)
    return template.strip() + "\n"


def _write_diff(root, step):
    base = f"step{step}-start"
    if not tag_exists(root, base):
        raise HarnessError(f"Step {step} has no builder work yet. Run `python harness.py session builder {step}` first.")
    excludes = [f":(exclude){d}" for d in C.DIFF_EXCLUDES]
    diff = git(root, "diff", f"{base}..HEAD", "--", ".", *excludes)
    if not diff.strip():
        raise HarnessError(
            f"There are no committed code changes since {base}. The builder must finish and commit "
            f"Step {step} before the reviewer starts.")
    d = state_dir(root) / "diffs"
    d.mkdir(exist_ok=True)
    path = d / f"step{step}_diff.patch"
    path.write_text(diff, encoding="utf-8")
    return base, path.relative_to(root).as_posix()


def start(root, role: str, step: int, launch: str = None, force: bool = False):
    if role not in C.ROLES:
        raise HarnessError(f"Role must be one of {', '.join(C.ROLES)}.")
    if step not in C.STEP_TITLES:
        raise HarnessError("Step must be 0-7.")
    if role not in C.roles_for_step(step):
        raise HarnessError("Step 0 has no tester. Use builder or reviewer.")
    if not has_commits(root):
        raise HarnessError("Run `python harness.py setup` first.")
    pending = load_session(root)
    if pending and not force:
        raise HarnessError(
            f"Session {pending['filename']} has not been saved yet. Run `python harness.py save` "
            f"(or `python harness.py session ... --force` to discard it).")

    values = {
        "STEP": str(step),
        "STEP_TITLE": C.STEP_TITLES[step],
        "DOC_REPORT": C.DOC_REPORTS.get(step, "PLAN.md"),
        "REVIEW_FILE": C.review_path(step),
        "TEST_LOG": C.test_log_path(step),
        "DIFF_FILE": "",
        "BASE": "",
        "FOCUS": "",
    }
    if role == "builder" and step > 0 and not tag_exists(root, f"step{step}-start"):
        git(root, "tag", f"step{step}-start")
        print(f"Tagged the current commit as step{step}-start (the reviewer will see changes from here).")
    if role == "reviewer" and step > 0:
        values["BASE"], values["DIFF_FILE"] = _write_diff(root, step)
    focus = C.REVIEWER_FOCUS if role == "reviewer" else C.TESTER_FOCUS if role == "tester" else {}
    values["FOCUS"] = "\n".join(f"- {q}" for q in focus.get(step, []))

    num = _next_number(root)
    today = dt.date.today().isoformat()
    filename = f"{num:02d}_{today}_{role}_step{step}"
    values["TRANSCRIPT"] = f"transcripts/{filename}.md"

    template = (root / "roles" / f"{role}.md").read_text(encoding="utf-8")
    prompt = _render(template, step, values)
    pdir = state_dir(root) / "prompts"
    pdir.mkdir(exist_ok=True)
    pfile = pdir / f"{filename}.prompt.md"
    pfile.write_text(prompt, encoding="utf-8")

    save_session(root, {
        "role": role, "step": step, "filename": filename,
        "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "head": head(root), "prompt_file": pfile.relative_to(root).as_posix(), "tool": launch,
    })

    print(f"\nSession: {role}, Step {step} ({C.STEP_TITLES[step]})")
    print(f"Transcript will be saved as: transcripts/{filename}.md")
    print(f"Prompt file: {pfile.relative_to(root).as_posix()}\n")

    if launch:
        exe = shutil.which(launch)
        if not exe:
            raise HarnessError(f"`{launch}` was not found on PATH. Start the session yourself and paste the prompt.")
        print(f"Starting a NEW {launch} session with the {role} prompt. Exit it when the work is done.\n")
        subprocess.call([exe, prompt], cwd=root)
        from .transcripts import save
        print()
        save(root, tool=launch)
    else:
        print("1. Start a NEW agent session (no earlier conversation) in this folder.")
        print("2. Paste the prompt below as the first message.")
        print("3. When the session is done, export its transcript and run:")
        print("     python harness.py save            (finds Claude Code or Codex sessions automatically)")
        print("     python harness.py save --file PATH (for an exported chat from any other tool)\n")
        print("-" * 72)
        print(prompt)
        print("-" * 72)
