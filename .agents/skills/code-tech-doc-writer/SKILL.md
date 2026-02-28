---
name: code-tech-doc-writer
description: Professional technical documentation writing expert for software projects. Use when user needs help with README, architecture docs, API docs, deployment docs, development guides, or code comments.
---

# Code Technical Documentation Writer

Write high-quality, well-structured technical documentation following industry standards.

## Principles

- **Reader-first**: Consider technical background and reading purpose
- **Concise**: Active voice, one core point per sentence
- **Example-driven**: Every concept with runnable code examples
- **Consistent**: Unified terminology and format throughout

## Document Types

### Type 1: README
Quick start for all users.

Key sections:
1. Badges (build, version, license)
2. One-line description
3. Features bullet list
4. Tech stack table
5. Quick start (install, config, run)
6. Project structure tree
7. Documentation links
8. License

See [references/readme-template.md](references/readme-template.md)

### Type 2: Architecture Design
System architecture and design decisions.

Key sections:
1. Document metadata (version, author, status)
2. Background and goals
3. Glossary
4. Architecture overview with Mermaid diagrams
5. Module division and interactions
6. Tech selection with ADRs
7. Data models and ER diagrams
8. Interface design
9. Non-functional design (performance, security, HA)
10. Deployment architecture
11. Risk assessment

See [references/architecture-template.md](references/architecture-template.md)

### Type 3: API Documentation
API specifications and reference.

Key sections:
1. Base URL and protocol
2. Authentication method
3. Common response format
4. Error codes table
5. Endpoint details (path, method, params, examples)
6. Pagination rules
7. Version strategy

See [references/api-doc-template.md](references/api-doc-template.md)

### Type 4: Development Guide
Local development and contribution guide.

Key sections:
1. Required tools and versions
2. Local setup steps
3. Project structure explanation
4. Code conventions (naming, comments, error handling)
5. Testing structure and examples
6. Commit message format
7. Code review checklist
8. Debugging tips
9. FAQ

See [references/dev-guide-template.md](references/dev-guide-template.md)

### Type 5: Deployment & Ops
System deployment and maintenance.

Key sections:
1. Deployment architecture diagram
2. Resource requirements table
3. Environment variables
4. Deployment procedures (first deploy, rolling update)
5. Monitoring and alerts
6. Log management
7. Backup and recovery
8. Troubleshooting guides
9. Security checklist
10. Maintenance schedule

See [references/deployment-template.md](references/deployment-template.md)

## Code Comment Standards

### File Header
```typescript
/**
 * @fileoverview Brief description
 * @description Detailed description
 * @module path/to/module
 * @author Name
 * @since 1.0.0
 */
```

### Function/Method
```typescript
/**
 * Brief description
 * 
 * @param paramName - Description
 * @returns Description
 * @throws {ErrorType} When/why thrown
 * @example
 * // code example
 */
```

### Class
```typescript
/**
 * Class description
 * 
 * @description Detailed capabilities
 * @implements {InterfaceName}
 */
```

## Style Guidelines

- **Active voice**: "系统返回结果" not "结果被系统返回"
- **Avoid fuzzy words**: "大概", "可能", "应该"
- **Keep English terms**: class, interface, method names in English
- **Define terms on first use**
- **Use Markdown tables** for structured data

## Workflow

When user asks for documentation:

1. **Identify document type** from the 5 types above
2. **Collect information**:
   - Tech stack and architecture
   - Core features
   - API definitions
   - Deployment environment
3. **Load appropriate template** from references/
4. **Fill with actual project info**:
   - Replace placeholders with real content
   - Provide runnable code examples
   - Add necessary diagrams
5. **Quality check**:
   - Content complete and accurate
   - Format consistent
   - Examples verifiable
   - Links accessible

## Quality Checklist

- [ ] Target audience clear
- [ ] Background info provided
- [ ] All features documented
- [ ] Code examples runnable
- [ ] Error handling explained
- [ ] Heading levels correct
- [ ] Tables formatted properly
- [ ] Code blocks have language tags
- [ ] Links are accessible
- [ ] Document date/version included
