import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { HumanMessage, SystemMessage, AIMessage, getBufferString } from '@langchain/core/messages';
import { getEncoding } from 'js-tiktoken';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
})

function countTokens(messages, encoder) {
    let total = 0;
    for (const msg of messages) {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
        total += encoder.encode(content).length;
    }
    return total;
}

async function summarizeHistory(messages) {
    if (messages.length === 0) return '';
    const conversationText = getBufferString(messages, {
        humanPrefix: '用户',
        aiPrefix: 'AI',
    });

    const summaryPrompt = `
    你是一个专业的总结专家，请总结以下核心内容，保留重要信息：
    ${conversationText}
    总结：
    `
    const summaryResponse = await model.invoke([new SystemMessage(summaryPrompt)]);
    return summaryResponse.content;
}

async function summarizationMemoryDemo() {
    const history = new InMemoryChatMessageHistory();
    const maxTokens = 100;
    const keepRecentTokens = 50;

    const encoder = getEncoding('cl100k_base');

    const messages = [
        { type: 'human', content: '你好，我叫李四' },
        { type: 'ai', content: '你好李四，很高兴认识你！' },
        { type: 'human', content: '我是一名设计师' },
        { type: 'ai', content: '设计师是个很有创造力的职业！你主要做什么类型的设计？' },
        { type: 'human', content: '我喜欢艺术和音乐' },
        { type: 'ai', content: '艺术和音乐都是很好的爱好，它们能激发创作灵感。' },
        { type: 'human', content: '我擅长 UI/UX 设计' },
        { type: 'ai', content: 'UI/UX 设计非常重要，好的用户体验能让产品更成功！' },
    ]

    for (const msg of messages) {
        if (msg.type === 'human') {
            await history.addMessage(new HumanMessage(msg.content));
        } else {
            await history.addMessage(new AIMessage(msg.content));
        }
    }

    let allMessages = await history.getMessages();
    const totalTokens = countTokens(allMessages, encoder);
    console.log(`原始消息数量: ${allMessages.length}`);
    console.log(`原始消息总Token数: ${totalTokens}`);
    console.log(`原始消息: ${allMessages.map(msg => msg.content).join('\n')}`);

    if (totalTokens > maxTokens) {
        const recentMessages = []
        let recentTokens = 0;

        for (let i = allMessages.length - 1; i >= 0; i--) {
            const msg = allMessages[i];
            const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
            const msgTokens = encoder.encode(content).length;

            if (recentTokens + msgTokens <= keepRecentTokens) {
                recentMessages.unshift(msg);
                recentTokens += msgTokens;
            } else {
                break
            }
        }

        const messagesToSummarize = allMessages.slice(0, allMessages.length - recentMessages.length);
        const summarizationTokens = countTokens(messagesToSummarize, encoder);
        console.log(`\nToken 数量过多，开始总结`)
        console.log(`将被总结的消息数量: ${summarizationTokens} Tokens`);
        console.log(`将被保留的消息数量: ${recentTokens} Tokens`);
        const summary = await summarizeHistory(messagesToSummarize);
        console.log(`\n 总结的内容: ${summary}`);

        await history.clear();
        for (const msg of recentMessages) {
            await history.addMessage(msg);
        }

        console.log(`\n 保留的消息数量: ${recentMessages.length}`);
        console.log(`\n 保留的消息: ${recentMessages.map(msg => {
            const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
            const tokens = encoder.encode(content).length;
            return `${msg.constructor.name}: ${content} (Tokens: ${tokens})`;
        }).join('\n')}`)
    }
}

summarizationMemoryDemo();