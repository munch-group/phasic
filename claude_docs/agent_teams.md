# Agent Teams Reference Guide

Reference for orchestrating teams of Claude Code sessions. Agent teams coordinate multiple independent Claude Code instances with shared tasks, inter-agent messaging, and centralized management.

**Source**: [code.claude.com/docs/en/agent-teams](https://code.claude.com/docs/en/agent-teams)

---

## When to Use Agent Teams vs Subagents

| Dimension | Subagents | Agent Teams |
|:----------|:----------|:------------|
| **Context** | Own context window; results return to caller | Own context window; fully independent |
| **Communication** | Report results back to main agent only | Teammates message each other directly |
| **Coordination** | Main agent manages all work | Shared task list with self-coordination |
| **Best for** | Focused tasks where only the result matters | Complex work requiring discussion and collaboration |
| **Token cost** | Lower: results summarized back to main context | Higher: each teammate is a separate Claude instance |

**Use subagents** when you need quick, focused workers that report back.
**Use agent teams** when teammates need to share findings, challenge each other, and coordinate on their own.

### Strong Use Cases for Teams

- **Research and review**: multiple teammates investigate different aspects simultaneously, then share and challenge findings
- **New modules or features**: teammates each own a separate piece without stepping on each other
- **Debugging with competing hypotheses**: teammates test different theories in parallel and converge faster
- **Cross-layer coordination**: changes spanning frontend, backend, and tests, each owned by a different teammate

### When NOT to Use Teams

- Sequential tasks with hard dependencies
- Same-file edits (causes overwrites)
- Routine tasks where coordination overhead exceeds benefit
- Work where a single session or subagents suffice

---

## Setup

### Enable Agent Teams

Agent teams are experimental and disabled by default. Enable via settings.json:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Or set the environment variable `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`.

Requires Claude Code v2.1.32 or later.

### Display Modes

Two display modes are available:

- **In-process** (default): all teammates run inside the main terminal. Use `Shift+Down` to cycle through teammates.
- **Split panes**: each teammate gets its own pane. Requires tmux or iTerm2.

Configure in `~/.claude.json`:

```json
{
  "teammateMode": "in-process"
}
```

Or per-session: `claude --teammate-mode in-process`

Default `"auto"` uses split panes if already in a tmux session, in-process otherwise.

---

## Architecture

| Component | Role |
|:----------|:-----|
| **Team lead** | Main session that creates the team, spawns teammates, coordinates work |
| **Teammates** | Separate Claude Code instances working on assigned tasks |
| **Task list** | Shared list of work items that teammates claim and complete |
| **Mailbox** | Messaging system for inter-agent communication |

### Storage Locations

- Team config: `~/.claude/teams/{team-name}/config.json` (auto-generated, do not edit by hand)
- Task list: `~/.claude/tasks/{team-name}/`

### Context and Communication

- Each teammate has its own context window
- Teammates load project context (CLAUDE.md, MCP servers, skills) but NOT the lead's conversation history
- Task-specific details must be included in the spawn prompt
- Messages are delivered automatically (no polling needed)
- Idle notifications sent to lead automatically
- All agents can see shared task list

### Permissions

- Teammates start with the lead's permission settings
- If lead uses `--dangerously-skip-permissions`, all teammates do too
- Individual modes can be changed after spawning but not at spawn time

---

## How to Use

### Starting a Team

Tell Claude to create a team with a description of the task and team structure:

```
I'm designing a CLI tool that helps developers track TODO comments across
their codebase. Create an agent team to explore this from different angles: one
teammate on UX, one on technical architecture, one playing devil's advocate.
```

### Specifying Teammates and Models

```
Create a team with 4 teammates to refactor these modules in parallel.
Use Sonnet for each teammate.
```

### Requiring Plan Approval

For complex or risky tasks, require teammates to plan before implementing:

```
Spawn an architect teammate to refactor the authentication module.
Require plan approval before they make any changes.
```

The teammate works in read-only plan mode until the lead approves. Rejected plans get feedback and the teammate revises. Influence approval criteria in your prompt:

```
Only approve plans that include test coverage.
Reject plans that modify the database schema.
```

### Talking to Teammates Directly

- **In-process mode**: `Shift+Down` to cycle, type to message, `Enter` to view session, `Escape` to interrupt, `Ctrl+T` for task list
- **Split-pane mode**: click into a teammate's pane

### Task Management

Tasks have three states: **pending**, **in progress**, **completed**.
Tasks can have dependencies (blocked until dependencies complete).

- **Lead assigns**: tell the lead which task to give to which teammate
- **Self-claim**: after finishing, a teammate picks up the next unassigned, unblocked task
- Task claiming uses file locking to prevent race conditions

### Using Subagent Definitions for Teammates

Reference a subagent type when spawning a teammate to reuse defined roles:

```
Spawn a teammate using the security-reviewer agent type to audit the auth module.
```

The teammate honors the definition's `tools` allowlist and `model`. Team coordination tools (SendMessage, task management) are always available regardless of `tools` restrictions.

Note: `skills` and `mcpServers` from subagent definitions are NOT applied to teammates. They load from project/user settings like a regular session.

### Shutting Down

Shut down individual teammates:
```
Ask the researcher teammate to shut down
```

Clean up the entire team (always do this from the lead):
```
Clean up the team
```

Shut down all teammates before cleaning up. Never let teammates run cleanup.

---

## Quality Gates with Hooks

Three hooks are available for enforcing rules:

| Hook | Trigger | Exit code 2 behavior |
|:-----|:--------|:---------------------|
| `TeammateIdle` | Teammate about to go idle | Sends feedback, keeps teammate working |
| `TaskCreated` | Task being created | Prevents creation, sends feedback |
| `TaskCompleted` | Task being marked complete | Prevents completion, sends feedback |

---

## Best Practices

### 1. Give Teammates Enough Context

Teammates don't inherit conversation history. Include all task-specific details in spawn prompts:

```
Spawn a security reviewer teammate with the prompt: "Review the authentication
module at src/auth/ for security vulnerabilities. Focus on token handling,
session management, and input validation. The app uses JWT tokens stored in
httpOnly cookies. Report any issues with severity ratings."
```

### 2. Choose Appropriate Team Size

- Start with **3-5 teammates** for most workflows
- Aim for **5-6 tasks per teammate**
- Three focused teammates often outperform five scattered ones
- Token costs scale linearly with team size

### 3. Size Tasks Appropriately

- **Too small**: coordination overhead exceeds benefit
- **Too large**: teammates work too long without check-ins
- **Just right**: self-contained units producing a clear deliverable (a function, a test file, a review)

### 4. Prevent the Lead from Doing Work Itself

If the lead starts implementing instead of delegating:
```
Wait for your teammates to complete their tasks before proceeding
```

### 5. Avoid File Conflicts

Break work so each teammate owns a different set of files. Two teammates editing the same file leads to overwrites.

### 6. Monitor and Steer

Check in on progress, redirect failing approaches, synthesize findings as they come in. Don't let a team run unattended too long.

### 7. Start with Research and Review

If new to agent teams, start with non-code tasks: reviewing PRs, researching libraries, investigating bugs. These show parallel value without coordination challenges.

---

## Effective Prompt Patterns

### Parallel Code Review

```
Create an agent team to review PR #142. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.
```

### Competing Hypothesis Debugging

```
Users report the app exits after one message instead of staying connected.
Spawn 5 agent teammates to investigate different hypotheses. Have them talk to
each other to try to disprove each other's theories, like a scientific
debate. Update the findings doc with whatever consensus emerges.
```

Key insight: the adversarial debate structure fights anchoring bias. Multiple independent investigators actively trying to disprove each other produce a much more reliable root cause than sequential investigation.

---

## Troubleshooting

| Problem | Solution |
|:--------|:---------|
| Teammates not appearing | Press `Shift+Down` to find them; check task complexity warrants a team |
| Too many permission prompts | Pre-approve common operations in permission settings before spawning |
| Teammates stopping on errors | Message them directly with new instructions, or spawn replacements |
| Lead shuts down too early | Tell it to keep going or wait for teammates to finish |
| Orphaned tmux sessions | `tmux ls` then `tmux kill-session -t <session-name>` |
| Split panes not working | Verify tmux is installed (`which tmux`) or iTerm2 Python API is enabled |

---

## Known Limitations

- No session resumption with in-process teammates (`/resume` and `/rewind` don't restore them)
- Task status can lag (teammates sometimes fail to mark tasks completed)
- Shutdown can be slow (teammates finish current request first)
- One team per session
- No nested teams (teammates cannot spawn their own teams)
- Lead is fixed for lifetime (no promotion or transfer)
- Permissions set at spawn (team-wide, changeable individually after)
- Split panes require tmux or iTerm2 (not supported in VS Code terminal, Windows Terminal, or Ghostty)
