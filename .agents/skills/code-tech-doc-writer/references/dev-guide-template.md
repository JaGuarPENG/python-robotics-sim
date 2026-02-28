# Development Guide Template

```markdown
# Development Guide

## 1. Environment Setup

### 1.1 Required Tools
| Tool | Version | Purpose |
|------|---------|---------|
| Node.js | >=18.0 | Runtime |
| npm | >=9.0 | Package manager |
| Git | >=2.30 | Version control |
| Docker | >=20.0 | Containerization |

### 1.2 Local Development Setup
```bash
# 1. Clone repository
git clone https://github.com/username/project.git
cd project

# 2. Install dependencies
npm install

# 3. Configure environment
cp .env.example .env
# Edit .env with necessary configs

# 4. Start database (Docker)
docker-compose up -d db

# 5. Run migrations
npm run migrate

# 6. Start dev server
npm run dev
```

## 2. Project Structure

### 2.1 Directory Layout
```
src/
├── config/           # Config files
│   ├── database.ts
│   └── app.ts
├── controllers/      # Controllers
├── services/         # Business logic
├── models/           # Data models
├── middlewares/      # Middleware
├── utils/            # Utilities
├── types/            # Type definitions
└── app.ts            # Entry point
```

### 2.2 Code Layers
- **Controller**: HTTP request/response handling
- **Service**: Business logic implementation
- **Model**: Data model definitions
- **Middleware**: Request preprocessing

## 3. Code Conventions

### 3.1 Naming
| Type | Convention | Example |
|------|------------|---------|
| Files | kebab-case | `user-service.ts` |
| Classes | PascalCase | `UserService` |
| Methods | camelCase | `getUserById` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRY` |

### 3.2 Comments
```typescript
/**
 * User service class
 * Provides user-related business logic
 */
class UserService {
  /**
   * Get user by ID
   * @param userId - User unique identifier
   * @returns User object or null if not found
   * @throws {NotFoundError} When user doesn't exist
   * @example
   * const user = await userService.getUserById('usr_123');
   */
  async getUserById(userId: string): Promise<User | null> {
    // Implementation
  }
}
```

### 3.3 Error Handling
```typescript
// Custom error class
class BusinessError extends Error {
  constructor(
    public code: string,
    message: string,
    public statusCode: number = 400
  ) {
    super(message);
    this.name = 'BusinessError';
  }
}

// Usage
try {
  const user = await userService.createUser(data);
  return res.json({ code: 201, data: user });
} catch (error) {
  if (error instanceof BusinessError) {
    return res.status(error.statusCode).json({
      code: error.statusCode,
      message: error.message
    });
  }
  logger.error('Unexpected error:', error);
  return res.status(500).json({ code: 500, message: 'Server error' });
}
```

## 4. Testing

### 4.1 Test Structure
```
tests/
├── unit/             # Unit tests
├── integration/      # Integration tests
├── e2e/              # End-to-end tests
└── fixtures/         # Test data
```

### 4.2 Test Example
```typescript
import { UserService } from '../../src/services/user.service';

describe('UserService', () => {
  let service: UserService;
  
  beforeEach(() => {
    service = new UserService();
  });
  
  describe('getUserById', () => {
    it('returns user when exists', async () => {
      const user = await service.getUserById('usr_123');
      expect(user).toBeDefined();
    });
    
    it('returns null when not found', async () => {
      const user = await service.getUserById('none');
      expect(user).toBeNull();
    });
  });
});
```

## 5. Commit Convention

### 5.1 Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 5.2 Types
| Type | Description |
|------|-------------|
| feat | New feature |
| fix | Bug fix |
| docs | Documentation |
| style | Code style |
| refactor | Refactoring |
| test | Testing |
| chore | Build/tools |

### 5.3 Example
```
feat(user): add user search

Implement user search by username and email:
- Support fuzzy matching
- Support pagination
- Add related tests

Closes #123
```

## 6. Code Review Checklist

- [ ] Code follows conventions
- [ ] Tests added
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance considered
- [ ] Error handling complete

## 7. Debugging

### 7.1 Logging
```typescript
import { logger } from '../utils/logger';

logger.debug('Debug info', { detail: 'value' });
logger.info('Normal info');
logger.warn('Warning');
logger.error('Error', error);
```

### 7.2 VSCode Debug Config
```json
{
  "version": "0.2.0",
  "configurations": [{
    "type": "node",
    "request": "launch",
    "name": "Debug Server",
    "program": "${workspaceFolder}/src/app.ts"
  }]
}
```

## 8. FAQ

### Q1: Install dependencies failed?
**A**: Clear cache and retry:
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Q2: Database connection failed?
**A**: Check Docker container:
```bash
docker-compose ps
docker-compose logs db
```
```
