import 'dotenv/config'
import Redis from 'ioredis'
import * as readline from 'node:readline/promises'

import { stdin, stdout } from 'node:process'
import { ChatOpenAI } from '@langchain/openai'
import { mapChatMessagesToStoredMessages, mapStoredMessagesToChatMessages } from '@langchain/core/messages'
import { createAgent, HumanMessage, summarizationMiddleware } from 'langchain'

const summaryPrompt = `你是对话摘要助手。请用中文总结以下对话，包含：
1. 讨论的主要话题
2. 用户提到的重要事实（姓名、偏好、日期等，务必保留原文信息）
3. 继续对话所需的关键上下文

保持简洁，不要编造，不要遗漏用户明确说过的信息。

待摘要的对话：
{messages}

摘要：`;

const sessionId = 'demo_user_001'

class RedisMessageRestore {
    redis = new Redis({
        host: 'localhost',
        port: 6379,
        db: 0
    })

    keyPrefix = 'agent:short_memory'

    ttlSeconds = 3600 // 1 小时

    constructor() {
        this.redis.on('connect', () => {
            console.log('Redis connected')
        })

        this.redis.on('error', (error) => {
            console.error('Redis error', error)
        })
    }

    messagesKey(sessionId) {
        return `${this.keyPrefix}:${sessionId}:messages`
    }

    async loadMessages(sessionId) {
        const raw = await this.redis.get(this.messagesKey(sessionId))
        if (!raw) {
            return []
        }
        return mapStoredMessagesToChatMessages(JSON.parse(raw))
    }

    async saveMessages(sessionId, messages) {
        const payload = JSON.stringify(mapChatMessagesToStoredMessages(messages))
        await this.redis.set(this.messagesKey(sessionId), payload, 'EX', this.ttlSeconds)
    }

    async clearMessages(sessionId) {
        await this.redis.del(this.messagesKey(sessionId))
    }

    async ttl(sessionId) {
        return await this.redis.ttl(this.messagesKey(sessionId))
    }
}

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
});

const agent = createAgent({
    model,
    tools: [],
    systemPrompt: '你是会话助手，记住用户提到的关键事实，中文简短回答，若消息中有对话摘要，请据此继续对话',
    middleware: [summarizationMiddleware({
        summaryPrompt,
        trigger: { messages: 3 },
        keep: { messages: 20 },
    })],
})

async function invokeWithMemory(agent, store, sessionId, userText) {
    const history = await store.loadMessages(sessionId);
    console.log(`  ↳ 从 Redis 加载 ${history.length} 条历史`);

    const result = await agent.invoke(
        { messages: [...history, new HumanMessage(userText)] },
        { recursionLimit: 30 },
    );

    await store.saveMessages(sessionId, result.messages);
    const ttl = await store.ttl(sessionId);
    console.log(`  ↳ 写回 Redis ${result.messages.length} 条 (TTL ${ttl}s)`);

    return result;
}

console.log("输入 exit / quit / :q 退出，:clear 清空记忆\n");

const store = new RedisMessageRestore()
const rl = readline.createInterface({ input: stdin, output: stdout })

let prevCount = (await store.loadMessages(sessionId)).length

try {
    while (true) {
        const userText = (await rl.question("你: ")).trim();
        if (!userText) continue;

        if (["exit", "quit", ":q"].includes(userText.toLowerCase())) break;

        if (userText === ":clear") {
            await store.clearMessages(sessionId);
            prevCount = 0;
            console.log("已清空当前会话记忆\n");
            continue;
        }

        const { messages } = await invokeWithMemory(agent, store, sessionId, userText);
        console.log("\n助手:", messages.at(-1)?.content);
        console.log(`当前消息数: ${messages.length}`);
        if (messages.length < prevCount + 2) {
            console.log("  ⚡ 已触发压缩");
        }
        prevCount = messages.length;
        console.log();
    }
} finally {
    rl.close();
}

await store.redis.quit();