# API Documentation Template

```markdown
# API Documentation

## 1. Overview

### 1.1 Basic Info
- Base URL: `https://api.example.com/v1`
- Protocol: HTTPS
- Data Format: JSON
- Encoding: UTF-8

### 1.2 Authentication
Use Bearer Token:
```http
Authorization: Bearer <your_access_token>
```

### 1.3 Common Response Format
```json
{
  "code": 200,
  "message": "success",
  "data": {},
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 1.4 Error Codes
| Code | Meaning | Solution |
|------|---------|----------|
| 400 | Bad Request | Check parameters |
| 401 | Unauthorized | Check token |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Check resource ID |
| 500 | Internal Error | Contact support |

## 2. Endpoints

### 2.1 Create Resource

#### Basic Info
- **URL**: `/resources`
- **Method**: POST
- **Description**: Create new resource

#### Request Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| name | string | Yes | Name, 3-50 chars |
| type | string | Yes | Resource type |
| tags | array | No | Tags |

#### Request Example
```json
{
  "name": "example",
  "type": "document",
  "tags": ["tag1", "tag2"]
}
```

#### Response Parameters
| Name | Type | Description |
|------|------|-------------|
| id | string | Unique ID |
| name | string | Name |
| created_at | string | Creation time |

#### Response Example
```json
{
  "code": 201,
  "message": "Created successfully",
  "data": {
    "id": "res_123456",
    "name": "example",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### Error Example
```json
{
  "code": 400,
  "message": "Validation error",
  "data": {
    "errors": [
      {"field": "name", "message": "Name is required"}
    ]
  }
}
```

### 2.2 Get Resource

#### Basic Info
- **URL**: `/resources/{id}`
- **Method**: GET
- **Description**: Get resource by ID

#### Path Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| id | string | Yes | Resource ID |

## 3. Pagination

### 3.1 Request Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| page | integer | No | 1 | Page number |
| page_size | integer | No | 20 | Items per page, max 100 |

### 3.2 Response Format
```json
{
  "code": 200,
  "data": {
    "items": [],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total": 100,
      "total_pages": 5
    }
  }
}
```

## 4. Versioning

### 4.1 Version Strategy
- URL path versioning: `/v1/`, `/v2/`
- Backward compatibility for minor versions
- 90-day deprecation notice

### 4.2 Changelog
| Version | Date | Changes |
|---------|------|---------|
| v1.1.0 | 2024-01-15 | Added search API |
| v1.0.0 | 2024-01-01 | Initial release |
```
