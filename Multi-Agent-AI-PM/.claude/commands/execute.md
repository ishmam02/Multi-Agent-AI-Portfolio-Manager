---
description: Execute implementation from a structured plan file
---

# Execute: Implement from Plan

## Plan to Execute

Read plan file: `$ARGUMENTS`

## Execution Instructions

### 1. Read and Understand

- Read the ENTIRE plan carefully
- Understand all tasks and their dependencies
- Note the validation commands to run
- Review the testing strategy
- Read ALL referenced codebase files listed in "Context References"
- Read ALL referenced documentation links

### 2. Execute Tasks in Order

For EACH task in the step-by-step task list:

#### a. Navigate to the task
- Identify the file and action required
- Read existing related files if modifying

#### b. Implement the task
- Follow the detailed specifications exactly
- Maintain consistency with existing code patterns
- Follow patterns referenced in the plan

#### c. Verify as you go
- After each file change, check syntax
- Ensure imports are correct
- Run the task's VALIDATE command if provided

### 3. Implement Testing Strategy

After completing implementation tasks:
- Create all test files specified in the plan
- Implement all test cases mentioned
- Follow the testing approach outlined
- Ensure tests cover edge cases listed

### 4. Run Validation Commands

Execute ALL validation commands from the plan in order:

**Level 1**: Syntax & Style (linters)
**Level 2**: Unit Tests
**Level 3**: Integration Tests
**Level 4**: Manual Validation steps

If any command fails:
- Fix the issue
- Re-run the command
- Continue only when it passes

### 5. Final Verification

Before completing:
- All tasks from plan completed
- All tests created and passing
- All validation commands pass
- Code follows project conventions

## Output Report

Provide summary:

### Completed Tasks
- List of all tasks completed
- Files created (with paths)
- Files modified (with paths)

### Tests Added
- Test files created
- Test cases implemented
- Test results

### Validation Results
```
# Output from each validation command
```

### Ready for Commit
- Confirm all changes are complete
- Confirm all validations pass
- Ready for `/commit` command

## Notes

- If you encounter issues not addressed in the plan, document them
- If you need to deviate from the plan, explain why
- If tests fail, fix implementation until they pass
- Don't skip validation steps
