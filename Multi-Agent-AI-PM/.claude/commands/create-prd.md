---
description: Generate a Product Requirements Document from conversation context
---

# Create PRD: Generate Product Requirements Document

## Output File

Write the PRD to: `$ARGUMENTS` (default: `PRD.md`)

## PRD Structure

Create a well-structured PRD with these sections. Adapt depth based on available information:

### Required Sections

**1. Executive Summary**
- Concise product overview (2-3 paragraphs)
- Core value proposition
- MVP goal statement

**2. Mission**
- Product mission statement
- Core principles (3-5 key principles)

**3. Target Users**
- Primary user personas
- Technical comfort level
- Key needs and pain points

**4. MVP Scope**
- **In Scope**: Core functionality (use checkboxes)
- **Out of Scope**: Deferred features (use checkboxes)

**5. User Stories**
- Primary user stories (5-8) in format: "As a [user], I want [action], so that [benefit]"
- Include concrete examples

**6. Core Architecture & Patterns**
- High-level architecture diagram (text)
- Directory structure
- Key design patterns

**7. Features**
- Detailed feature specifications
- API designs if applicable

**8. Technology Stack**
- All technologies with versions and purpose

**9. Security & Configuration**
- Auth approach
- Environment variables
- Security scope

**10. API Specification** (if applicable)
- Endpoints, request/response formats

**11. Success Criteria**
- Functional requirements (checkboxes)
- Quality indicators
- UX goals

**12. Implementation Phases**
- 3-6 phases, each with: Goal, Deliverables, Validation criteria
- Each phase = one PIV loop

**13. Future Considerations**
- Post-MVP enhancements

**14. Risks & Mitigations**
- 3-5 key risks with mitigations

**15. Appendix**
- Key dependencies with doc links

## Instructions

### 1. Extract Requirements
- Review entire conversation history
- Identify explicit requirements and implicit needs
- Note technical constraints and preferences

### 2. Synthesize
- Organize into sections
- Fill reasonable assumptions where details are missing
- Ensure consistency

### 3. Write the PRD
- Clear, professional language
- Concrete examples over abstract descriptions
- Markdown formatting (headings, lists, code blocks, tables, checkboxes)

### 4. Quality Checks
- All sections present and complete
- User stories have clear benefits
- MVP scope is realistic
- Implementation phases are actionable
- Success criteria are measurable

## After Creating

1. Confirm file path
2. Brief summary of contents
3. Highlight any assumptions made
4. Suggest next steps (review, then `/create-rules`)
