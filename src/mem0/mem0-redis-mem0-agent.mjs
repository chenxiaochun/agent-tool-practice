/**
 * ============================================================================
 * 新手导读：这份文件在干什么？
 * ============================================================================
 *
 * 你要做一个「会聊天、会记住事」的 Agent。人的记忆分两种：
 * - 短期记忆：刚才聊了什么（几分钟～几十分钟内还记得）
 * - 长期记忆：你是谁、住哪、有什么过敏（过很久还记得）
 *
 * 本文件用两套存储对应这两种记忆：
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  Redis（短时）                                                  │
 * │  存：当前会话的完整对话记录（你说一句、助手回一句……）            │
 * │  特点：有过期时间 TTL；同一会话内「上一句说了啥」靠它            │
 * ├─────────────────────────────────────────────────────────────────┤
 * │  Mem0（长时，还分两层）                                         │
 * │  · user 层：跨会话的长期事实（姓名、过敏、偏好）                 │
 * │             只绑 userId → 换一个聊天窗口还能认出来你             │
 * │  · session 层：仅本会话的任务/进度（「这次写 Q1 总结」）         │
 * │             绑 userId + runId(sessionId) → 换会话就不应带过来   │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * 为什么要分这么细？
 * - 全塞进模型上下文：聊久了 token 爆炸、又贵又慢
 * - 全塞 Redis：重启换会话后「你是谁」会丢
 * - 全塞 Mem0：寒暄、临时数字也会污染长期记忆
 * 所以：刚说过的话 → Redis；值得长期/本会话记住的事实 → Mem0（先分类再写）
 *
 * ----------------------------------------------------------------------------
 * 用户敲下一句话之后，数据怎么流？（对应下面的 invokeWithMemory）
 * ----------------------------------------------------------------------------
 *  1. 从 Redis 取出「之前聊过的消息列表」
 *  2. 用这句话去 Mem0 里「搜索」相关记忆（user 层 + session 层一起搜）
 *  3. 把搜到的记忆写成一条 SystemMessage（系统提示），和历史拼在一起
 *  4. 交给 Agent / 大模型生成回答（消息太多时中间件会先摘要压缩）
 *  5. 把新的对话历史写回 Redis（并去掉 SystemMessage，见 messagesForRedis）
 *  6. 另请一个「分类器」模型判断：要不要写入 Mem0？写到哪一层？
 *  7. 把助手回复打印给你看
 *
 * 阅读建议：先看 invokeWithMemory（主链路），再回头看两个 Store 类的细节。
 * ============================================================================
 */

import "dotenv/config";
import Redis from "ioredis";
import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { z } from "zod";
import { MemoryClient } from "mem0ai";
import { ChatOpenAI } from "@langchain/openai";
import {
    SystemMessage,
    SystemMessageChunk,
    HumanMessage,
    mapChatMessagesToStoredMessages,
    mapStoredMessagesToChatMessages,
} from "@langchain/core/messages";
import { createAgent, summarizationMiddleware } from "langchain";

// ============================================================================
// 配置：从环境变量读，没有就用默认值（?? 表示「左边是 null/undefined 才用右边」）
// ============================================================================

/** Redis 连哪台机器、哪个库。本地开发一般是 localhost:6379 */
const REDIS_HOST = process.env.REDIS_HOST ?? "localhost";
const REDIS_PORT = Number(process.env.REDIS_PORT ?? 6379);
const REDIS_DB = Number(process.env.REDIS_DB ?? 0);

/**
 * MEMORY_TTL：短时记忆存活秒数。默认 1800 = 30 分钟。
 * 含义：超过这段时间没人再聊，Redis 会自动删掉这个会话的历史。
 * 每次 saveMessages 都会用 SET ... EX 把倒计时重新拨满（续期）。
 */
const MEMORY_TTL = Number(process.env.MEMORY_TTL_SECONDS ?? 1800);

/** Redis key 前缀，避免和别的业务 key 撞名 */
const KEY_PREFIX = process.env.MEMORY_KEY_PREFIX ?? "agent:short_memory";

/**
 * USER_ID / SESSION_ID 是「记忆的门牌号」：
 * - USER_ID：你是哪个用户（长期记忆挂在这个人身上）
 * - SESSION_ID：你开的哪一次聊天窗口（会话层记忆挂在「人 + 这次会话」上）
 *
 * 类比：USER_ID = 身份证号；SESSION_ID = 某天某次进店办业务的小票号。
 */
const USER_ID = process.env.MEM0_USER_ID ?? "demo_user_001";
const SESSION_ID = "session_002";

/** 每次从 Mem0 最多取回几条最相关记忆（太大浪费 token，太小可能漏关键事实） */
const MEM0_TOP_K = Number(process.env.MEM0_TOP_K ?? 5);

/**
 * memorySchema：分类结果的形状（用 zod 校验模型吐出的 JSON）。
 *
 * 三个字段：
 * - write_user    → 要不要写进「长期用户画像」
 * - write_session → 要不要写进「当前会话便签」
 * - reason        → 为什么这样判（方便你看日志学习）
 *
 * 注意：通义等兼容接口对 withStructuredOutput / json_schema 支持不稳定，
 * 本示例改为「普通聊天 + 要求输出 JSON + zod 校验」，更稳。
 */
const memorySchema = z.object({
    write_user: z.boolean(),
    write_session: z.boolean(),
    reason: z.string(),
});

/**
 * CLASSIFIER_PROMPT：分类器的「阅卷标准 / 工作手册」。
 * 它不负责陪你聊天，只负责看完这一轮对话后，勾选 write_user / write_session。
 * 规则写得越清楚，越少把「这次的任务」误存成「一辈子的画像」。
 */
const CLASSIFIER_PROMPT = `
你是记忆分层分类器。判断本轮对话是否有「新事实」需写入 Mem0，并分到正确层级。
你必须只输出一个 JSON 对象，不要 Markdown，不要其它解释。格式严格如下：
{"write_user":true或false,"write_session":true或false,"reason":"一句话理由"}

## user 层（跨会话长期）
- 用户身份与画像：姓名、职业、居住地、长期爱好
- 长期偏好与约束：饮食过敏、回答风格、常用技术栈
- 持续数周以上的个人背景（非单次任务）

## session 层（仅当前会话）
- 当前正在做的任务、目标、文档大纲、方案草稿
- 本会话内的进度、决策、待办、临时约定
- 用户明确用「这次」「本轮」「当前会话」描述的工作上下文

## 均不写入
- 寒暄、致谢、纯确认
- 助手生成的通用内容（攻略、示例代码、建议清单），用户未明确采纳为新事实
- 无信息增量的复述

## 决策原则
1. 「这次我们先写 Q1 总结」「当前在排查 XX」→ 优先 session，不要标成 user
2. 仅含身份/住址/长期爱好/过敏/持久偏好，且没有「这次」「本轮」「当前任务/会话」等会话限定 → write_user=true，write_session 必须为 false
   （反例：不要把「我叫小明，住在杭州，喜欢骑行」同时写入 session）
3. user 与 session 可同时为 true，仅当同一轮里既有跨会话长期事实、又有明确的本会话任务/进度
4. 一次性请求（如「帮我做旅行攻略」）且未产生需跨轮记住的约定 → 均为 false
5. 拿不准时：宁可少写 session，也不要把纯画像误标成会话任务
`;

/** 从模型回复里抠出 JSON（兼容偶发的 ```json 包裹） */
function parseClassifierJson(text) {
    const raw = String(text ?? "").trim();
    const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
    const candidate = fenced ? fenced[1].trim() : raw;
    const start = candidate.indexOf("{");
    const end = candidate.lastIndexOf("}");
    if (start < 0 || end < start) {
        throw new Error(`分类器未返回 JSON: ${raw.slice(0, 200)}`);
    }
    return memorySchema.parse(JSON.parse(candidate.slice(start, end + 1)));
}

/**
 * summaryPrompt：给「摘要中间件」用的提示词。
 *
 * 当 Redis 里消息越积越多，模型上下文装不下时，中间件会把旧消息压成一段摘要。
 * 这里特意写：长期偏好交给 Mem0，摘要只抓「会话里正在干的事」，避免两套记忆重复堆砌。
 * {messages} 是占位符，运行时会被中间件替换成真正要摘要的消息。
 */
const summaryPrompt = `
    你是对话摘要助手。用中文简洁总结：话题、会话内进度/报错/待办。
    用户级长期偏好由外部记忆维护，摘要勿重复堆砌。不要编造。

    待摘要的对话：
    {messages}

    摘要：
`;

/**
 * messagesForRedis：写 Redis 之前的「安检门」。
 *
 * LangChain 消息大致分几类：
 * - HumanMessage：用户说的话
 * - AIMessage：助手说的话
 * - SystemMessage：系统指令 / 我们注入的记忆提示（不是用户真实说过的话）
 * - SystemMessageChunk：流式场景下的系统消息碎片
 *
 * 为什么过滤掉 SystemMessage？
 * 1. Agent 的 systemPrompt、Mem0 记忆提示，每次 invoke 都会由代码重新组装
 * 2. 若存进 Redis，下一轮又会和新建的 SystemMessage 叠在一起 → 重复、混乱、浪费空间
 * 3. Redis 只应保存「真实对话轨迹」，方便下一轮接着聊
 *
 * filter 返回 true 的元素会留下：即「不是 SystemMessage 且不是 SystemMessageChunk」的消息。
 */
function messagesForRedis(messages) {
    return messages.filter(
        (m) => !SystemMessage.isInstance(m) && !SystemMessageChunk.isInstance(m),
    );
}

// ============================================================================
// RedisMessageStore：短时对话历史的增删查
// 可以把它想成「这个聊天窗口的草稿纸」，有自动销毁倒计时。
// ============================================================================

/**
 * 存储格式：一个 Redis 字符串 key → 一整段 JSON（消息数组）。
 * key 例子：agent:short_memory:session_002:messages
 *
 * 为什么用 JSON 字符串而不是 Redis List？
 * 本示例消息量不大，整包读写更简单；真正上线高并发可以再换更细的结构。
 */
class RedisMessageStore {
    constructor({ redis, keyPrefix, ttlSeconds }) {
        this.redis = redis;         // ioredis 客户端
        this.keyPrefix = keyPrefix; // key 前缀
        this.ttlSeconds = ttlSeconds; // 每次写入时设置的过期秒数
    }

    /** 把 sessionId 拼成完整 Redis key，保证不同会话互不覆盖 */
    messagesKey(sessionId) {
        return `${this.keyPrefix}:${sessionId}:messages`;
    }

    /**
     * 加载历史消息
     * 1. GET key → 拿到字符串（可能是 null）
     * 2. 没有数据 → 返回空数组（表示新会话）
     * 3. JSON.parse 成普通对象数组
     * 4. mapStoredMessagesToChatMessages：还原成 LangChain 的 Message 实例
     *    （因为 Redis 只能存文本，不能直接存 JS 类实例）
     */
    async loadMessages(sessionId) {
        const raw = await this.redis.get(this.messagesKey(sessionId));
        if (!raw) return [];
        return mapStoredMessagesToChatMessages(JSON.parse(raw));
    }

    /**
     * 保存历史消息
     * 1. mapChatMessagesToStoredMessages：Message 实例 → 可 JSON 化的纯数据
     * 2. JSON.stringify 成字符串
     * 3. SET key value EX ttl：写入并设置过期时间
     *
     * 注意：每次保存都会重置 TTL。也就是说只要你还在聊，草稿就不会过期；
     * 停聊超过 MEMORY_TTL 秒，Redis 会自动清空这轮会话历史。
     */
    async saveMessages(sessionId, messages) {
        const payload = JSON.stringify(mapChatMessagesToStoredMessages(messages));
        await this.redis.set(this.messagesKey(sessionId), payload, "EX", this.ttlSeconds);
    }

    /** 立刻删掉该会话草稿（对应终端命令 :clear） */
    async clear(sessionId) {
        await this.redis.del(this.messagesKey(sessionId));
    }

    /**
     * 查还剩多少秒过期（Redis TTL 命令）
     * - 返回正数：还剩 N 秒
     * - 返回 -1：key 存在但没有过期时间（本示例正常不会出现，因为我们总带 EX）
     * - 返回 -2：key 根本不存在
     * 用来在日志里确认「写回成功且续期生效」。
     */
    async ttl(sessionId) {
        return this.redis.ttl(this.messagesKey(sessionId));
    }
}

// ============================================================================
// Mem0MemoryStore：长期 / 会话事实的检索、注入、分类写入
// 可以把它想成「通讯录 + 本次工单便签」，由云端 Mem0 服务托管。
// ============================================================================

/**
 * 和 RedisMessageStore 的差别：
 * - Redis 存的是「原话流水账」（messages 列表）
 * - Mem0 存的是「提炼后的事实条目」（如「用户对海鲜过敏」），可按语义搜索
 */
class Mem0MemoryStore {
    constructor({ client, userId, sessionId, topK, classifier }) {
        this.client = client;         // mem0ai 的 MemoryClient
        this.userId = userId;         // 用户门牌
        this.sessionId = sessionId;   // 会话门牌（写入时作为 runId）
        this.topK = topK;             // 检索条数上限
        this.classifier = classifier; // 负责「这轮该不该记、记哪层」的结构化 LLM
    }

    /**
     * 语义搜索：用当前用户这句话当 query，找出相关记忆。
     *
     * 为什么查两次？
     * - 第一次只要 user_id：拿到「这个人是谁」的长期画像
     * - 第二次 user_id AND run_id：拿到「这次会话在干嘛」的便签
     * 两层不要混成一次查询，否则难以在提示词里分区展示，也难排查。
     *
     * Promise.all：两次网络请求并行，总耗时≈较慢的那一次，而不是相加。
     *
     * 返回形状：{ user: [...], session: [...] }
     * 每条结果里通常有 .memory 字段（人类可读的一句话事实）。
     */
    async search(query) {
        const [userRes, sessionRes] = await Promise.all([
            // 1) 用户长期层
            this.client.search(query, {
                filters: { user_id: this.userId },
                topK: this.topK,
            }),
            // 2) 当前会话层（必须同时匹配用户和会话）
            this.client.search(query, {
                filters: {
                    AND: [{ user_id: this.userId }, { run_id: this.sessionId }],
                },
                topK: this.topK,
            }),
        ]);
        return {
            user: userRes.results ?? [],
            session: sessionRes.results ?? [],
        };
    }

    /**
     * 把「检索到的事实条目」翻译成模型能读的系统提示。
     *
     * 新手易混点：Mem0 返回的是数据对象，模型不会自动看到它们。
     * 你必须显式放进 messages 数组里（这里做成 SystemMessage）。
     *
     * 步骤：
     * 1. 若有 user 结果 → 拼「【用户长期记忆】」小节
     * 2. 若有 session 结果 → 拼「【当前会话记忆】」小节
     * 3. 两节都空 → 返回 null（调用方就不要往 messages 里塞）
     * 4. 否则合并成一条 SystemMessage，并叮嘱「勿编造」
     */
    buildSystemMessage({ user, session }) {
        const blocks = [];
        if (user.length) {
            blocks.push(`【用户长期记忆】\n${user.map((m) => `- ${m.memory}`).join("\n")}`);
        }
        if (session.length) {
            blocks.push(`【当前会话记忆】\n${session.map((m) => `- ${m.memory}`).join("\n")}`);
        }
        if (!blocks.length) return null;
        return new SystemMessage(`${blocks.join("\n\n")}\n\n请结合以上记忆回答，勿编造。`);
    }

    /**
     * 一轮对话结束后：决定要不要把新事实写入 Mem0。
     *
     * 为什么不「每轮无脑 add」？
     * - 「你好」「谢谢」写进去会污染记忆库
     * - 临时数字、助手随口建议也可能被当成用户事实
     * 所以先让分类器看一轮对话，再按需写入。
     *
     * 步骤：
     * 1. 把本轮原文收成 Mem0 要求的 messages 格式（role + content）
     * 2. 调用 classifier（带 memorySchema）→ 得到 write_user / write_session / reason
     * 3. write_user === true  → add，只带 userId（跨会话可见）
     * 4. write_session === true → add，带 userId + runId（仅本会话可见）
     * 5. 返回 written（实际写了哪些层）和 reason（给人看的解释）
     *
     * 注意：两层可以同时为 true（例如一句话里既说了职业，又说了这次任务）。
     */
    async classifyAndPersist(userText, assistantText) {
        // 1. Mem0 add 的输入不是随便一个字符串，而是「对话片段」数组
        const turn = [
            { role: "user", content: userText },
            { role: "assistant", content: assistantText },
        ];

        // 2. 分类：普通聊天输出 JSON，再本地解析（兼容通义等网关）
        const classifyMsg = await this.classifier.invoke([
            new SystemMessage(CLASSIFIER_PROMPT),
            new HumanMessage(
                `请根据下面这轮对话输出 JSON。\n用户：${userText}\n助手：${assistantText}`,
            ),
        ]);
        const { write_user, write_session, reason } = parseClassifierJson(
            classifyMsg.content,
        );

        // 3–4. 按开关写入（Mem0 云端可能异步处理，刚写入立刻 search 有时还是空的，属正常）
        const written = [];
        if (write_user) {
            await this.client.add(turn, { userId: this.userId });
            written.push("user");
        }
        if (write_session) {
            await this.client.add(turn, { userId: this.userId, runId: this.sessionId });
            written.push("session");
        }

        // 5.
        return { written, reason };
    }

    /**
     * 清空 Mem0 记忆（对应终端 :clear-mem0）
     * 1. deleteAll({ userId }) → 清该用户的长期层
     * 2. deleteAll({ userId, runId }) → 清该用户当前会话层
     * 两步都要做，否则可能只清掉一半。
     */
    async clear() {
        await this.client.deleteAll({ userId: this.userId });
        await this.client.deleteAll({ userId: this.userId, runId: this.sessionId });
    }
}

// ============================================================================
// invokeWithMemory：单轮对话的「总导演」
// 把上面两个 Store + Agent 串成一条完整流水线。看懂这个函数，就看懂本文件 80%。
// ============================================================================

/**
 * 参数说明：
 * - agent       ：会说话的 Agent（内部是大模型 + 可选中间件）
 * - redisStore  ：短时历史仓库
 * - mem0Store   ：长期/会话事实仓库
 * - sessionId   ：当前聊天窗口 ID
 * - userText    ：用户刚刚输入的那一句话
 *
 * 返回：
 * - messages       ：Agent 产出的完整消息列表（含可能的 SystemMessage）
 * - redisMessages  ：真正写进 Redis 的消息（已过滤 SystemMessage）
 * - assistantText  ：最后一条助手回复的纯文本，方便打印
 */
async function invokeWithMemory(agent, redisStore, mem0Store, sessionId, userText) {
    // ---------- 1. 读短时历史 ----------
    // 没有历史时是 []，Agent 就当第一轮聊。
    const history = await redisStore.loadMessages(sessionId);
    console.log(`  ↳ Redis 加载 ${history.length} 条历史`);

    // ---------- 2. 检索长期 / 会话记忆 ----------
    // 用「当前这句话」做语义检索：问过敏时更容易命中过敏相关记忆。
    const mem = await mem0Store.search(userText);
    if (mem.user.length) console.log(`  ↳ Mem0 用户层 ${mem.user.length} 条`);
    if (mem.session.length) console.log(`  ↳ Mem0 会话层 ${mem.session.length} 条`);

    // ---------- 3. 组装本轮喂给 Agent 的 messages ----------
    // 顺序很重要，可以记成「先交代背景，再回顾前文，最后才是本句」：
    //   [可选：Mem0 记忆 SystemMessage] + [Redis 历史...] + [当前 HumanMessage]
    //
    // ...(memoryMsg ? [memoryMsg] : []) 的意思：
    //   有记忆就展开成一个元素；没记忆就展开成空，不往数组里塞 undefined。
    const memoryMsg = mem0Store.buildSystemMessage(mem);
    const invokeMessages = [
        ...(memoryMsg ? [memoryMsg] : []),
        ...history,
        new HumanMessage(userText),
    ];

    // ---------- 4. 调用 Agent ----------
    // recursionLimit：防止工具循环调用把自己绕死（本示例 tools 为空，但仍建议保留）。
    // 若历史很长，summarizationMiddleware 会在模型真正调用前先压缩 messages。
    const result = await agent.invoke(
        { messages: invokeMessages },
        { recursionLimit: 30 },
    );

    // ---------- 5. 写回 Redis 短时记忆 ----------
    // 重要：寒暄、闲聊也一样会写 Redis！
    // Redis 管的是「刚才聊了什么」（短时对话流水），不是「值不值得长期记住」。
    // 「要不要进 Mem0」是下一步分类器的事；和 Redis 无关。
    //
    // result.messages 通常 = 输入消息 + 新的 AI 回复（若发生摘要，旧消息可能已被替换成摘要）。
    // 先过滤 SystemMessage，再整包覆盖写回，并刷新 TTL。
    const redisMessages = messagesForRedis(result.messages);
    const dropped = result.messages.length - redisMessages.length;
    await redisStore.saveMessages(sessionId, redisMessages);
    const ttl = await redisStore.ttl(sessionId);
    console.log(
        `  ↳ Redis 写回 ${redisMessages.length} 条` +
        (dropped ? `（过滤 ${dropped} 条 SystemMessage）` : "") +
        ` (TTL ${ttl}s)`,
    );

    // ---------- 6. 分类并写入 Mem0（失败不影响 Redis 已保存的结果） ----------
    // 取最后一条消息当助手回复；再用分类器决定要不要沉淀成事实。
    // 寒暄通常会被判为「均不写入」→ 日志出现「Mem0 未写入」，这是预期，不等于 Redis 没写。
    const assistantText = String(result.messages.at(-1)?.content ?? "");
    let written = [];
    let reason = "";
    try {
        ({ written, reason } = await mem0Store.classifyAndPersist(userText, assistantText));
        console.log(`  ↳ 分类: ${reason}`);
        console.log(written.length ? `  ↳ Mem0 写入: ${written.join(", ")}` : "  ↳ Mem0 未写入（短时对话仍在 Redis）");
    } catch (err) {
        // 分类器 / Mem0 出错时：对话已经在 Redis 里了，不能让整轮聊天白做
        console.error(`  ↳ Mem0 分类/写入失败（Redis 已保存）: ${err.message}`);
    }

    // ---------- 7. 交给外层 REPL 展示 ----------
    return { messages: result.messages, redisMessages, assistantText };
}

// ============================================================================
// 程序启动区（从上到下执行一次）
// 可以想成餐厅开业：查执照 → 接通水电 → 摆好灶台 → 开门迎客
// ============================================================================

// ---------- 1. 检查必备环境变量 ----------
// MEM0_API_KEY：Mem0 云服务密钥；OPENAI_API_KEY：大模型密钥（也可是兼容网关）
if (!process.env.MEM0_API_KEY || !process.env.OPENAI_API_KEY) {
    console.error("需要 MEM0_API_KEY 与 OPENAI_API_KEY");
    process.exit(1);
}

// ---------- 2. 创建底层客户端 ----------
const redis = new Redis({ host: REDIS_HOST, port: REDIS_PORT, db: REDIS_DB });
const mem0 = new MemoryClient({ apiKey: process.env.MEM0_API_KEY });

// 监听连接事件：方便你在终端确认 Redis 是否活着
redis.on("connect", () => console.log("✅ Redis 已连接"));
redis.on("error", (err) => console.error("❌ Redis 错误:", err.message));

// ---------- 3. 启动前探活 ----------
// ping 不通就直接退出，避免后面每次请求都报一长串难懂的错
try {
    await redis.ping();
} catch {
    console.error("Redis 未连接，请先执行: docker compose up -d redis");
    process.exit(1);
}

// ---------- 4. 包一层 Redis 仓库（业务代码不要到处写 redis.get/set） ----------
const redisStore = new RedisMessageStore({
    redis,
    keyPrefix: KEY_PREFIX,
    ttlSeconds: MEMORY_TTL,
});

/** 对话模型与分类器共用的连接配置；temperature: 0 让输出更稳、更少随机发挥 */
const llmOpts = {
    apiKey: process.env.OPENAI_API_KEY,
    configuration: { baseURL: process.env.OPENAI_BASE_URL },
    temperature: 0,
};

// ---------- 5. 主对话模型（真正陪你聊天的那个） ----------
const model = new ChatOpenAI({ model: process.env.MODEL_NAME, ...llmOpts });

// ---------- 6. 分类器模型 ----------
// 不用 withStructuredOutput：通义等兼容接口经常返回空对象，导致 zod 报
// write_user/write_session/reason 为 undefined，进而永远走不到 Mem0 add。
// 改为普通 ChatOpenAI，靠提示词要 JSON，再用 parseClassifierJson + zod 校验。
const classifier = new ChatOpenAI({ model: process.env.MODEL_NAME, ...llmOpts });

// ---------- 7. 包一层 Mem0 仓库 ----------
const mem0Store = new Mem0MemoryStore({
    client: mem0,
    userId: USER_ID,
    sessionId: SESSION_ID,
    topK: MEM0_TOP_K,
    classifier,
});

/**
 * ---------- 8. 创建 Agent ----------
 *
 * createAgent 帮你把「模型 + 系统提示 + 中间件」组装成可 invoke 的对象。
 *
 * summarizationMiddleware 参数怎么理解：
 * - trigger: { messages: 8 }
 *     当消息条数达到约 8 条，就「尝试」做摘要（想压缩）。
 * - keep: { messages: 4 }
 *     压缩时尽量保留最近约 4 条原文，更早的内容收成摘要。
 *
 * 新手注意：若 keep 设得比当前消息数还大，实际可能压不动
 * （中间件发现没东西可砍就会跳过）。演示压缩时 keep 要小于你预期的历史长度。
 */
const agent = createAgent({
    model,
    tools: [], // 本示例不挂工具；以后可加搜索、查库等
    systemPrompt:
        "你是会话助手。结合系统消息中的长期/会话记忆回答，中文简短。有对话摘要则据此继续。",
    middleware: [
        summarizationMiddleware({
            model,
            summaryPrompt,
            trigger: { messages: 8 },
            keep: { messages: 4 },
        }),
    ],
});

console.log(`用户 ${USER_ID} | 会话 ${SESSION_ID}`);
console.log("输入 exit / quit / :q 退出；:clear 清空 Redis；:clear-mem0 清空 Mem0\n");

// ---------- 9. 打开终端问答循环（REPL） ----------
const rl = readline.createInterface({ input: stdin, output: stdout });

/**
 * prevCount：上一轮结束时 Redis 里有多少条消息。
 * 用来粗略判断「这一轮有没有触发摘要压缩」：
 * - 正常一轮：至少多 2 条（新的 human + 新的 ai）→ 长度 ≈ prevCount + 2
 * - 若触发压缩：旧消息被摘要替换，总条数可能 < prevCount + 2
 * 这不是 100% 精确的官方信号，但对学习演示够用。
 */
let prevCount = (await redisStore.loadMessages(SESSION_ID)).length;

try {
    // 无限循环：一直读用户输入，直到 break
    while (true) {
        const userText = (await rl.question("你: ")).trim();
        if (!userText) continue; // 空回车就再问一次

        // 9.1 退出命令
        if (["exit", "quit", ":q"].includes(userText.toLowerCase())) break;

        // 9.2 只清短时记忆：长期 Mem0 还在。适合「这段对话重来，但还记得我是谁」
        if (userText === ":clear") {
            await redisStore.clear(SESSION_ID);
            prevCount = 0;
            console.log("已清空 Redis 短期记忆\n");
            continue;
        }

        // 9.3 清 Mem0：长期画像和本会话便签都丢掉。适合重新做记忆实验
        if (userText === ":clear-mem0") {
            await mem0Store.clear();
            console.log("已清空 Mem0 用户层与当前会话层\n");
            continue;
        }

        // 9.4 正常聊天：走进总导演 invokeWithMemory
        const { redisMessages, assistantText } = await invokeWithMemory(
            agent,
            redisStore,
            mem0Store,
            SESSION_ID,
            userText,
        );

        console.log("\n助手:", assistantText);
        console.log(`Redis 消息数: ${redisMessages.length}`);
        if (redisMessages.length < prevCount + 2) {
            console.log("  ⚡ 已触发压缩");
        }
        prevCount = redisMessages.length;
        console.log();
    }
} finally {
    // 无论正常退出还是中途抛错，都关掉 readline，避免进程挂住
    rl.close();
}

// ---------- 10. 释放 Redis 连接，让进程优雅退出 ----------
await redis.quit();

/*
 * ============================================================================
 * 动手实验脚本（建议先 :clear-mem0 再 :clear，从干净状态开始）
 * ============================================================================
 *
 * 一、寒暄
 * 你好 / 在吗 / 谢谢
 * → 预期：Mem0 未写入。说明分类器会挡掉无信息量内容。
 *
 * 二、自我介绍（练 user 层）
 * 我叫小明，住在杭州，平时喜欢骑行和摄影。
 * 我对海鲜过敏，出差尽量别安排沿海城市。
 * → 预期：Mem0 写入 user。换会话 / 重启后仍应能答「你是谁 / 有什么过敏」。
 *
 * 三、当前任务（练 session 层）
 * 这次我们先写 Q1 季度总结，大纲分三块：项目复盘、数据指标、下季度计划。
 * 项目复盘里重点写 order-service 的 500 错误排查过程。
 * → 预期：Mem0 写入 session。清 mem0 或换 sessionId 后不应再带这些任务细节。
 *
 * 四、一句话里两层都有
 * 我长期做后端开发，这次会话的任务是排查 payment-api 超时，先从 P99 日志看起。
 * 另外我之后技术回答都希望带代码示例，这个一直记住。
 * → 预期：可能同时写 user + session。看日志里的「分类: …」学习模型怎么切分。
 *
 * 五、Redis vs Mem0
 * 刚才说的 payment-api，超时阈值先假设 3 秒。
 * 上一句我说的阈值是多少？
 * → 预期：靠 Redis 短时历史就能答。不必也不应依赖 Mem0。
 *
 * 重启进程（不要 :clear-mem0）再问：我是谁？有什么过敏？
 * → 预期：Redis 可能空了，但 user 层仍在，说明长期记忆生效。
 *
 * 六、触发摘要压缩（可选）
 * 连续聊超过约 8 条消息（trigger），观察是否打印「已触发压缩」。
 * → 理解：短时历史过长时，中间件会用摘要换空间。
 *
 * 推荐体验顺序：
 * 清空 → 寒暄 → 自我介绍 → 当前任务 → 重启验 user → :clear-mem0 验 session 消失
 */
