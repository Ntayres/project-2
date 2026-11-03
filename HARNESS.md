# Project 2 Harness

This harness runs the builder → reviewer → tester → fix cycle required by ASSIGNMENT.md and keeps the record your grade depends on. It:

- writes a role prompt for each new agent session, so the reviewer and tester start with fresh context;
- gives the reviewer only the code diff, not the builder's explanations;
- saves each session's transcript under the required name;
- logs every test run in the step's test log;
- rejects commit messages without a role tag;
- shows your progress and checks your submission before you hand it in.

It works with any agentic coding tool. With Claude Code or Codex CLI it can also start the session and save the transcript for you.

## One-time setup

```
pip install -r requirements.txt
python harness.py setup
```

Copy the dataset files into `input_data/` (they are not committed). Then fill in the `[TODO]` parts of AGENTS.md and commit them with a `[student]` tag.

Requires Python 3.9+ and git.

## The cycle for each step

| # | Who | Command or action | Produces |
| --- | --- | --- | --- |
| 1 | Builder | `python harness.py session builder N` | Code in src/, docs/ report, `[builder]` commits |
| 2 | You | `python harness.py save`, then commit the transcript | transcripts/NN_date_builder_stepN.md |
| 3 | Reviewer | `python harness.py session reviewer N` | reviews/stepN_review.md (comments) |
| 4 | You | `python harness.py save`, then commit | transcript |
| 5 | You | Fill in the Decision log in the review: Accepted or Rejected, and why | `[student]` commit |
| 6 | Tester | `python harness.py session tester N` | tests/test_stepN_*.py, tests/stepN_test_log.md |
| 7 | You | `python harness.py save`, then commit | transcript |
| 8 | You | Record each failing test's decision in the test log | `[student]` commit |
| 9 | Builder | `python harness.py session builder N` again (fix mode) | Fixes for accepted comments and failing tests |
| 10 | You | `python harness.py test N` until it passes, then commit the log | New run in the test log |

Step 0 (the plan) is the same without the tester: the builder writes PLAN.md, the reviewer reviews it, you add decisions, and the builder revises the plan. Finish Step 0 before any code.

Run `python harness.py status` at any time to see what is done and what to do next.

## Starting a session

`python harness.py session ROLE STEP` prints a prompt. Then:

1. Start a **new** session in your tool, with no earlier conversation, in this folder.
2. Paste the prompt as the first message.
3. Let the agent work. Steer it if needed; your messages are part of the record.
4. When it is done, close the session and save the transcript (below).

To have the harness start the tool for you:

```
python harness.py session builder 1 --launch claude
python harness.py session reviewer 1 --launch codex
```

The transcript is saved automatically when you exit.

## Saving transcripts

| Tool | How to save |
| --- | --- |
| Claude Code | `python harness.py save` finds the session log automatically. |
| Codex CLI | `python harness.py save --tool codex` |
| Cursor, Copilot, others | Export or copy the whole chat to a file, then `python harness.py save --file path/to/chat.md` |

Each save writes `transcripts/NN_YYYY-MM-DD_role_stepN.md` (and the raw `.jsonl` when there is one). Do not edit transcripts, except to remove API keys, passwords, or personal information. Commit each transcript with the tag of the role that made it, e.g., `[reviewer] Step 1 transcript`.

## Commit tags

| Tag | For |
| --- | --- |
| `[builder]` | Code, PLAN.md, docs/ reports |
| `[reviewer]` | Review reports |
| `[tester]` | Tests and test logs |
| `[student]` | AGENTS.md, decision logs, REPORT.md, REFLECTION.md, README.md |

A git hook rejects any other message. Commit after each task; never squash or rewrite history.

## Commands

| Command | What it does |
| --- | --- |
| `python harness.py setup` | Creates the git repo, installs the commit hook, checks packages, tags the start |
| `python harness.py session ROLE STEP [--launch claude\|codex]` | Starts a builder, reviewer, or tester session for Step 0-7 |
| `python harness.py save [--file PATH] [--tool claude\|codex]` | Saves the current session's transcript |
| `python harness.py test STEP` | Runs tests/test_stepSTEP_*.py and adds the result to the test log |
| `python harness.py test all` | Runs every test (not logged) |
| `python harness.py status` | Shows progress for each step and the next action |
| `python harness.py check [--out FILE]` | Checks the whole submission |

## Before you submit

Run `python harness.py check`. Fix every `[FAIL]`. Each `[WARN]` is something a grader will look at; fix it or be ready to explain it.

## Files

| Path | Written by |
| --- | --- |
| AGENTS.md, CLAUDE.md | You (CLAUDE.md just points Claude Code to AGENTS.md) |
| ASSIGNMENT.md | Instructor (the tester and reviewer read it) |
| PLAN.md | Builder, Step 0 |
| src/ | Builder |
| docs/stepN_*.md | Builder |
| reviews/stepN_review.md | Reviewer (comments) and you (decision log) |
| tests/test_stepN_*.py, tests/stepN_test_log.md | Tester (you add failure decisions; the harness adds runs) |
| transcripts/ | The harness, when you save a session |
| README.md, REPORT.md, REFLECTION.md | You |
| roles/, templates/, harness/, hooks/ | The harness. Do not change these. |

---

## For instructors

Check a student's repository:

```
python harness.py check --repo path/to/student-repo --out student_check.md
```

The checker reports PASS, WARN, or FAIL for:

- **Required files:** all 27 required Markdown files present; no `[TODO]` placeholders left.
- **Review reports:** every comment has an Accepted or Rejected decision and a reason; at least one rejection overall.
- **Testing:** test files for every step; logged runs; latest run passing; a decision for every failure.
- **Role separation:** a transcript for every role and step; no transcript contains another role's prompt; no agent session id shared between roles (Claude Code and Codex logs).
- **Git history:** every commit has a role tag; commits per role; total commit count; very large commits; plan written, reviewed, and revised before the first code commit.
- **Consistency:** role-tagged commits have a transcript of the same role within a day; commit hashes cited in decision logs exist; REFLECTION.md cites transcripts or commits.

A WARN is a prompt to look closer, not proof of a problem. The checker cannot prove that a transcript is genuine; the live check-in is still the strongest test. Run the checker from your own copy of the harness with `--repo`, not from the student's copy, since students can edit theirs.
