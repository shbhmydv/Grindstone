# Security Policy

## The trust model (read this before running Grindstone)

Grindstone is an orchestrator that **executes code on your behalf**. By design it
runs, as the invoking user:

1. **Shell commands chosen by the planner.** A job's `done_when` / `exit_criterion`
   / `complete_run` `evidence` checks are shell commands (`CmdCheck`). They come
   from the planner (a cloud model, GPT-5.5 via `codex exec`) steered by your
   `job.md`. Grindstone runs them via a shell to compute a deterministic pass/fail.
2. **The request scripts named in the target repo's config.** `.grindstone/config.yaml`
   names the `models/*.sh` adapters for each role; these are `exec`'d. To blunt the
   clone-and-run risk, a configured `script:` path **must resolve under the bundled
   `models/` directory**. Set `GRINDSTONE_ALLOW_REPO_SCRIPTS=1` to opt a *trusted*
   repo's own scripts back in.
3. **The optional final-polish pass edits your repo.** When `final_polish` is
   configured, `codex` runs in `workspace-write` mode against a throwaway worktree.
   Its edits are **kept only if the run's evidence still passes**, are committed to
   a branch but **never auto-pushed** (you approve all pushes), and network access
   is pinned off.

### What this means for you

- **Run Grindstone only on job specs and target repos you trust.** Treat the
  planner's output as untrusted code: a malicious or careless `job.md`, or a
  compromised planner endpoint, can produce a check command that does anything your
  user account can.
- **Prefer a disposable VM or container** for untrusted or experimental work.
- Do not point Grindstone at a repository you would not run `make` / `npm install`
  in; its `.grindstone/config.yaml` and its checks are part of its attack surface.

## What is sandboxed / gated

- **Run-dir paths** (keyed-log references, artifact names, `vision_review`
  screenshots) are validated to stay inside the run dir / eval worktree
  (path-traversal and symlink-escape guards).
- **Worker integration** happens in isolated git worktrees with file-ownership
  scoping; the deterministic checks, not model claims, gate every phase and the run.
- **The final-polish pass** cannot escape its worktree into your working tree, is
  evidence-gated, and is never auto-pushed.
- Credentials (codex / opencode / pi auth) live in those tools' own config outside
  this repo; Grindstone never reads or logs them.

## Reporting a vulnerability

Please report security issues privately by opening a
[GitHub security advisory](https://docs.github.com/code-security/security-advisories)
on the repository, or by email to the maintainer, rather than a public issue.
We will acknowledge and triage as quickly as we can.
