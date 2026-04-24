---
description: "Create comprehensive feature plan with deep codebase analysis and research"
---

# Plan a new task

## Feature: $ARGUMENTS

## Mission

Transform a feature request into a **comprehensive implementation plan** through systematic codebase analysis, external research, and strategic planning.

**Core Principle**: We do NOT write code in this phase. Our goal is to create a context-rich implementation plan that enables one-pass implementation success for AI agents.

**Key Philosophy**: Context is King. The plan must contain ALL information needed for implementation — patterns, mandatory reading, documentation, validation commands — so the execution agent succeeds on the first attempt.

## Planning Process

### Phase 1: Feature Understanding

**Deep Feature Analysis:**
- Extract the core problem being solved
- Identify user value and business impact
- Determine feature type: New Capability / Enhancement / Refactor / Bug Fix
- Assess complexity: Low / Medium / High
- Map affected systems and components

**Create User Story:**
```
As a <type of user>
I want to <action/goal>
So that <benefit/value>
```

### Phase 2: Codebase Intelligence Gathering

**Use specialized subagents for parallel analysis:**

**1. Project Structure Analysis**
- Map directory structure and architectural patterns
- Identify service/component boundaries
- Locate configuration files
- Find environment setup and build processes

**2. Pattern Recognition**
- Search for similar implementations in codebase
- Identify coding conventions (naming, structure, error handling, logging)
- Extract common patterns for the feature's domain
- Check CLAUDE.md for project-specific rules

**3. Dependency Analysis**
- Catalog external libraries relevant to feature
- Find relevant documentation in reference/ or docs/
- Note library versions and compatibility

**4. Testing Patterns**
- Identify test framework and structure
- Find similar test examples for reference
- Note coverage requirements

**5. Integration Points**
- Identify existing files that need updates
- Determine new files to create and their locations
- Map API registration patterns
- Understand database/model patterns if applicable

**Clarify Ambiguities:**
- If requirements are unclear, ask the user to clarify before continuing
- Get specific implementation preferences
- Resolve architectural decisions before proceeding

### Phase 3: External Research & Documentation

**Use subagents for external research:**
- Research latest library versions and best practices
- Find official documentation with specific section anchors
- Locate implementation examples
- Identify common gotchas and known issues

### Phase 4: Strategic Thinking

- How does this feature fit into the existing architecture?
- What are the critical dependencies and order of operations?
- What could go wrong? (Edge cases, race conditions, errors)
- How will this be tested comprehensively?
- Are there security considerations?

### Phase 5: Plan Generation

Write the plan to: `.agents/plans/{kebab-case-feature-name}.md`

Create `.agents/plans/` directory if it doesn't exist.

**Use this template:**

```markdown
# Feature: <feature-name>

Read these files before implementing. Pay attention to naming of existing utils, types, and models.

## Feature Description
<Detailed description>

## User Story
As a <user> I want to <goal> so that <benefit>

## Problem Statement
<The specific problem this addresses>

## Solution Statement
<Proposed approach>

## Feature Metadata
- **Type**: [New Capability/Enhancement/Refactor/Bug Fix]
- **Complexity**: [Low/Medium/High]
- **Systems Affected**: [list]
- **Dependencies**: [external libs/services]

---

## CONTEXT REFERENCES

### Files to Read Before Implementing
- `path/to/file.py` (lines X-Y) — Why: <reason>

### New Files to Create
- `path/to/new_file.py` — <purpose>

### Documentation to Read
- [Doc Link](url#section) — Why: <reason>

### Patterns to Follow
<Code examples extracted from codebase>

---

## STEP-BY-STEP TASKS

Execute every task in order. Each task is atomic and independently testable.

### Task 1: {ACTION} {target_file}
- **IMPLEMENT**: {what to do}
- **PATTERN**: {reference to existing pattern — file:line}
- **VALIDATE**: `{executable command}`

### Task 2: ...

---

## TESTING STRATEGY

### Unit Tests
<scope and requirements>

### Integration Tests
<scope and requirements>

### Edge Cases
<specific edge cases to test>

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style
<linting commands>

### Level 2: Unit Tests
<test commands>

### Level 3: Integration Tests
<test commands>

### Level 4: Manual Validation
<manual steps>

---

## ACCEPTANCE CRITERIA
- [ ] All specified functionality implemented
- [ ] All validation commands pass
- [ ] Tests cover edge cases
- [ ] Code follows project conventions
- [ ] No regressions

## COMPLETION CHECKLIST
- [ ] All tasks completed in order
- [ ] All validation commands pass
- [ ] Full test suite passes
- [ ] Ready for /commit
```

## Output

After creating the plan, report:
- Summary of feature and approach
- Full path to plan file
- Complexity assessment
- Key risks
- Confidence score (#/10) for one-pass implementation success
