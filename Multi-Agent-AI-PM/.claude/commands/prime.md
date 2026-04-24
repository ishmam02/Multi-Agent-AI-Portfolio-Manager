---
description: Prime agent with codebase understanding at the start of every session
---

# Prime: Load Project Context

## Objective

Build comprehensive understanding of the codebase by analyzing structure, documentation, and key files. Run this at the start of every new session before any planning or implementation.

## Process

### 1. Analyze Project Structure

List all tracked files:
!`git ls-files`

Show directory structure:
!`find . -type f -not -path '*/node_modules/*' -not -path '*/__pycache__/*' -not -path '*/.git/*' -not -path '*/dist/*' -not -path '*/build/*' -not -path '*/.venv/*' -not -path '*/.mypy_cache/*' | head -100`

### 2. Read Core Documentation

- Read PRD.md (what we're building)
- Read CLAUDE.md (how we build it — global rules)
- Read README.md if it exists
- Read any architecture docs in reference/
- Read .agent-config.yml for repo-specific setup (linters, CI, test framework)
- Read pyproject.toml or package.json for dependencies

### 3. Identify Key Files

Based on the structure, identify and read:
- Main entry points (main.py, index.ts, app.py, etc.)
- Core configuration files
- Key model/schema definitions
- Important service or controller files
- Test configuration

### 4. Understand Current State

Check recent activity:
!`git log -15 --oneline`

Check current branch and status:
!`git status`

Check what's been worked on recently:
!`git log -5 --stat`

### 5. Identify Next Phase

Based on the PRD implementation phases:
- Determine which phases are complete (from git log and existing code)
- Identify the next phase to work on
- Note any incomplete work or blockers

## Output Report

Provide a concise summary covering:

### Project Overview
- Purpose and type of application
- Current version/state

### Tech Stack
- Languages, frameworks, key libraries

### Architecture
- Overall structure and organization
- Key patterns identified

### Current State
- Active branch
- Recent changes (from git log)
- What's been built so far

### Next Up
- Recommended next phase/feature from PRD
- Any prerequisites or blockers

**Keep this scannable — bullet points and clear headers. This is context for the next step, not documentation.**
