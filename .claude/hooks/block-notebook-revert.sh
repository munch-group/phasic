#!/usr/bin/env bash
# PreToolUse(Bash) hook — notebook-safety guardrail.
#
# Blocks any git command that could DISCARD uncommitted changes
# (checkout / restore / clean / stash / reset --hard) *when* there are
# uncommitted Jupyter notebook (.ipynb) changes in the working tree, because
# those changes are unrecoverable once overwritten. Added after a
# `git checkout -- docs/` accidentally destroyed uncommitted notebook bug fixes.
#
# Receives the tool call as JSON on stdin; emits a PreToolUse deny decision
# (as JSON on stdout) only when both conditions hold. Otherwise it is a no-op.
set -euo pipefail

cmd=$(jq -r '.tool_input.command // ""' 2>/dev/null || true)

# 1) Does the command invoke a git op that can throw away working-tree changes?
if printf '%s' "$cmd" | grep -qE '\bgit\b.*(\bcheckout\b|\brestore\b|\bclean\b|\bstash\b|reset[[:space:]]+--hard)'; then
  # 2) Are there uncommitted .ipynb changes that such a command could destroy?
  if git status --porcelain 2>/dev/null | grep -q '[.]ipynb'; then
    jq -n '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: "BLOCKED by notebook-safety hook: this git command can discard uncommitted changes, and there ARE uncommitted .ipynb changes in the working tree that would be unrecoverable. NEVER revert or overwrite uncommitted notebooks. Re-scope the revert to explicit non-notebook paths (e.g. `git checkout -- docs/api docs/cpp_api`), never a whole directory or `.`. If touching notebooks is truly intended, ask the user first."
      }
    }'
  fi
fi
exit 0
