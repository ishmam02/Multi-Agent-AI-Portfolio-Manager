---
description: Create a standardized git commit for all uncommitted changes
---

# Commit: Standardized Git Commit

## Process

### 1. Review Changes

Run these commands to understand what's being committed:

!`git status`
!`git rev-parse HEAD > /dev/null 2>&1 && git diff HEAD || git diff --cached`
!`git status --porcelain`

### 2. Stage Changes

Add all untracked and changed files that are relevant to the current work.
Do NOT stage files containing secrets (.env, credentials, API keys).

### 3. Create Commit

Create an atomic commit with a descriptive message following this format:

```
<tag>: <concise description of what changed and why>

<optional body with more detail if needed>
```

**Tags:**
- `feat`: New feature or capability
- `fix`: Bug fix
- `refactor`: Code restructuring without behavior change
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `chore`: Build, config, dependency changes
- `style`: Formatting, linting fixes

**Guidelines:**
- Keep the first line under 72 characters
- Focus on WHAT changed and WHY, not HOW
- Use imperative mood ("add X" not "added X")
- Be specific enough that git log serves as long-term memory
