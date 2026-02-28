---
name: generate-tdd
description: Generate internal technical design documents (TDD) by analyzing codebase structure and business logic. Use when user asks for technical documentation, architecture analysis, code-level design documents, or generating TDD/PRD documents from existing code.
---

# Generate Technical Design Document

Analyze codebase and generate professional technical design documents following top-tier tech company standards.

## Workflow

### Phase 1: Codebase Analysis
1. Start from entry points (main.py, app.py, index.js, etc.)
2. Identify architecture patterns (MVC, microservices, etc.)
3. Extract data models and ORM definitions
4. Trace core business logic flows

### Phase 2: Architecture Synthesis
1. Define system boundaries and external dependencies
2. Map key business use cases end-to-end
3. Identify non-functional requirements (caching, auth, circuit breakers)

### Phase 3: Document Generation
Follow the output structure in [references/output-template.md](references/output-template.md).

Use Mermaid.js for diagrams:
- `C4Context` or `graph TD` for system context
- `sequenceDiagram` for core business flows

### Phase 4: Review
- Verify all code references are accurate
- Ensure no hallucinated features
- Check diagrams render correctly

## Rules

- Keep all class names, variables, and file paths in original English
- All architecture inferences must be based on actual code
- Use Markdown tables for API routes, env vars, and DB schemas
- Document body must be in professional Chinese (mainland tech style)
- If a feature is not found in code, explicitly state: "> 检查当前代码库上下文，尚未发现..."

## Output Structure

See [references/output-template.md](references/output-template.md) for the complete document template.
