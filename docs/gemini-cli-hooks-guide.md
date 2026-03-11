# Gemini CLI 钩子使用文档

## 什么是钩子

钩子是在 Gemini CLI 执行过程中特定时刻运行的脚本，用来拦截和自定义行为。

## 常用管理命令

```
/hooks panel              # 查看所有钩子状态
/hooks enable <名称>       # 启用指定钩子
/hooks disable <名称>      # 禁用指定钩子
/hooks enable-all         # 启用所有钩子
/hooks disable-all        # 禁用所有钩子
```

## 钩子事件类型

| 事件 | 触发时机 | 用途 |
|------|----------|------|
| `SessionStart` | 会话开始时 | 初始化 |
| `BeforeAgent` | 用户提交提示后 | 添加上下文 |
| `BeforeTool` | 工具执行前 | 验证/拦截操作 |
| `AfterTool` | 工具执行后 | 记录/处理结果 |
| `BeforeModel` | 发送给 LLM 前 | 修改提示 |
| `AfterModel` | LLM 响应后 | 处理响应 |
| `SessionEnd` | 会话结束时 | 清理 |

## 配置文件

编辑 `.gemini/settings.json`：

```json
{
  "hooks": {
    "BeforeTool": [
      {
        "matcher": "write_file",
        "hooks": [
          {
            "type": "command",
            "command": "bash .gemini/hooks/check.sh",
            "name": "my-hook",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

### 字段说明

- `matcher`: 匹配条件（正则表达式，如 `"write_file|replace"`）
- `type`: 执行类型（目前只支持 `"command"`）
- `command`: 要执行的命令
- `name`: 钩子名称（用于 `/hooks enable/disable`）
- `timeout`: 超时时间（毫秒）

## 编写钩子脚本

脚本通过 `stdin` 接收 JSON，通过 `stdout` 返回 JSON。

### 输入示例

```json
{
  "tool_name": "write_file",
  "tool_input": {
    "path": "test.js",
    "content": "console.log('hello')"
  }
}
```

### 输出示例

**允许操作：**
```json
{"decision": "allow"}
```

**阻止操作：**
```json
{
  "decision": "deny",
  "reason": "原因",
  "systemMessage": "显示给用户的消息"
}
```

**添加上下文：**
```json
{
  "hookSpecificOutput": {
    "hookEventName": "BeforeAgent",
    "additionalContext": "额外上下文信息"
  }
}
```

### 简单示例

```bash
#!/bin/bash
# 读取输入
input=$(cat)

# 日志输出到 stderr
echo "执行钩子" >&2

# 返回结果到 stdout
echo '{"decision": "allow"}'
```

## 实用示例

### 1. 阻止敏感信息写入

```bash
#!/bin/bash
input=$(cat)
content=$(echo "$input" | jq -r '.tool_input.content // ""')

if echo "$content" | grep -qi "password"; then
  echo '{"decision": "deny", "reason": "包含敏感词"}'
  exit 0
fi

echo '{"decision": "allow"}'
```

配置：
```json
{
  "hooks": {
    "BeforeTool": [
      {
        "matcher": "write_file",
        "hooks": [
          {
            "type": "command",
            "command": "bash .gemini/hooks/block-secret.sh",
            "name": "secret-check"
          }
        ]
      }
    ]
  }
}
```

### 2. 自动添加 Git 上下文

```bash
#!/bin/bash
commits=$(git log -3 --oneline 2>/dev/null)

cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "BeforeAgent",
    "additionalContext": "最近提交：\n$commits"
  }
}
EOF
```

配置：
```json
{
  "hooks": {
    "BeforeAgent": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash .gemini/hooks/git-context.sh",
            "name": "git-context"
          }
        ]
      }
    ]
  }
}
```

### 3. 记录工具执行

```bash
#!/bin/bash
input=$(cat)
tool=$(echo "$input" | jq -r '.tool_name')
echo "[$(date)] $tool" >> .gemini/hooks.log
echo '{"decision": "allow"}'
```

## 注意事项

1. **同步执行**：钩子会阻塞，保持脚本简单快速
2. **stdout 只能输出 JSON**：日志用 `>&2` 输出到 stderr
3. **退出码**：0 表示成功，2 表示阻止
4. **权限**：钩子以用户权限运行，审查后再启用

## 快速检查清单

- [ ] 脚本有执行权限：`chmod +x hook.sh`
- [ ] 配置中 `name` 字段已填写
- [ ] `matcher` 正确匹配目标工具
- [ ] 输出是合法 JSON
- [ ] 日志输出到 stderr（`>&2`）
