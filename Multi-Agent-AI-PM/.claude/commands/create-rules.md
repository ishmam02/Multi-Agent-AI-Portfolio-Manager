---
description: Generate CLAUDE.md global rules by analyzing the codebase
---

# Create Global Rules

Generate a CLAUDE.md file by analyzing the codebase and extracting patterns.

## Objective

Create project-specific global rules that give Claude context about:
- What this project is
- Technologies used
- How the code is organized
- Patterns and conventions to follow
- How to build, test, and validate

## Phase 1: Discover

### Identify Project Type
- Web App (Full-stack, Frontend, Backend)
- API/Backend
- Library/Package
- CLI Tool
- Orchestration/Infrastructure

### Analyze Configuration
Look at root config files: pyproject.toml, package.json, tsconfig.json, etc.

### Map Directory Structure
Explore the codebase to understand organization.

## Phase 2: Analyze

### Extract Tech Stack
From config files, identify: runtime, framework, database, testing, build tools, linting.

### Identify Patterns
Study existing code for: naming conventions, file organization, error handling, logging, types/interfaces.

### Find Key Files
Entry points, configuration, core business logic, shared utilities, type definitions.

## Phase 3: Generate

Use the template at `.claude/CLAUDE-template.md` as a starting point if it exists.

**Output**: `CLAUDE.md` at project root.

**Key sections:**

1. **Project Overview** — what is this and what does it do?
2. **Tech Stack** — technologies with purpose
3. **Commands** — dev, build, test, lint commands
4. **Project Structure** — directory layout with descriptions
5. **Architecture** — data flow, key patterns
6. **Code Patterns** — naming conventions, file organization, error handling
7. **Testing** — framework, patterns, how to run
8. **Validation** — commands to run before committing
9. **Key Files** — important files to know about
10. **On-Demand Context** — reference docs for deeper context (progressive disclosure)

## Phase 4: Create On-Demand Context

If the project warrants it, create reference docs in `.claude/reference/` or as skills:
- `reference/api.md` — API endpoint patterns
- `reference/components.md` — frontend component patterns
- `reference/testing.md` — testing patterns and fixtures

Point to these from the "On-Demand Context" section in CLAUDE.md.

## Tips

- Keep CLAUDE.md under 250 lines — concise and scannable
- Don't duplicate info that's in other docs (link instead)
- Focus on patterns and conventions, not exhaustive documentation
- Update as the project evolves (system evolution mindset)
