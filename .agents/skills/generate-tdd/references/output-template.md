# TDD Output Template

Use this structure when generating technical design documents.

## 1. 系统边界与工程上下文

### 1.1 系统范围 (Scope)
- 核心领域逻辑
- 主要功能边界

### 1.2 非目标 (Non-Goals)
- 明确排除的功能
- 剥离给其他服务的职责

### 1.3 关键假设与依赖
| 依赖项 | 版本/要求 | 说明 |
|--------|-----------|------|
| | | |

## 2. 宏观架构拓扑

### 2.1 C4 系统上下文图
```mermaid
C4Context
    title System Context Diagram
    System_Boundary(system, "系统") {
        // define components
    }
```

### 2.2 架构模式
描述采用的架构模式及技术考量。

## 3. 核心组件设计与 API 契约

### 3.1 模块划分
| 模块 | 职责 | 关键技术 |
|------|------|----------|
| | | |

### 3.2 API 契约
| 端点 | 方法 | 认证 | 描述 |
|------|------|------|------|
| | | | |

### 3.3 核心业务链路时序
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Service
    participant DB
    // define flow
```

## 4. 领域数据模型与持久化策略

### 4.1 存储机制选型
- 数据库类型及驱动
- ORM/查询方式

### 4.2 核心实体映射
| 实体 | 核心字段 | 外键/索引 | 说明 |
|------|----------|-----------|------|
| | | | |

## 5. 非功能性需求实现 (NFRs)

### 5.1 安全防护体系
- 认证授权机制
- 敏感数据处理

### 5.2 性能优化
- 缓存策略
- 并发模型

### 5.3 可观测性
- 日志规范
- 监控埋点

## 6. 第三方依赖与技术债

### 6.1 核心依赖树
| 库名 | 版本 | 用途 |
|------|------|------|
| | | |

### 6.2 技术债评估
- 潜在风险点
- 重构建议
