---
# 示例片段（从范本提炼）

完整风格见：`src/mem0/mem0-redis-mem0-agent.mjs`

## 导读里的主流程序号

```text
 *  1. 从 Redis 取出历史
 *  2. 去 Mem0 搜索相关记忆
 *  3. 拼成 SystemMessage
 *  4. 交给 Agent 生成回答
 *  5. 写回 Redis
 *  6. 分类器决定是否写入 Mem0
 *  7. 打印回复
```

## 函数注释 + 行内步骤对齐

```js
/**
 * 步骤：
 * 1. 组装 turn
 * 2. 调用 classifier
 * 3. 按需写入 user 层
 * 4. 按需写入 session 层
 * 5. 返回 written 与 reason
 */
async classifyAndPersist(userText, assistantText) {
    // 1. …
    const turn = [/* … */];
    // 2. …
    const result = await this.classifier.invoke(/* … */);
    // 3–4. …
    // 5.
    return { written, reason };
}
```

## 装配阶段横幅序号

```js
// ---------- 5. 主对话模型 ----------
// ---------- 6. 分类器模型 ----------
// ---------- 7. 包一层 Mem0 仓库 ----------
// ---------- 8. 创建 Agent ----------
// ---------- 9. 打开终端问答循环 ----------
    // 9.1 退出
    // 9.2 清 Redis
    // 9.3 清 Mem0
    // 9.4 正常聊天
// ---------- 10. 释放连接 ----------
```

## 「为什么」短句

```js
/**
 * 为什么过滤掉 SystemMessage？
 * 1. 每次 invoke 都会重新组装
 * 2. 若存进 Redis 会和下轮新建的叠在一起
 * 3. Redis 只应保存真实对话轨迹
 */
```
