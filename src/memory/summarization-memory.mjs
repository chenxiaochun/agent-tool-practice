import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { HumanMessage, SystemMessage, AIMessage, getBufferString } from '@langchain/core/messages';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
})

async function summarizationHistory(messages) {
    if (messages.length === 0) return '';
    const conversationText = getBufferString(messages, {
        humanPrefix: '用户',
        aiPrefix: 'AI',
    });

    const summaryPrompt = `
       请总结以下核心内容，保留重要信息：
       ${conversationText}

       总结：
    `
    const summaryResponse = await model.invoke([new SystemMessage(summaryPrompt)]);
    return summaryResponse.content;
}


async function summarizationMemoryDemo() {
    const history = new InMemoryChatMessageHistory();
    const maxMessages = 6;
    const messages = [
        { type: 'human', content: '我叫李四' },
        { type: 'ai', content: '你好李四，很高兴认识你！' },
        { type: 'human', content: '我是一名设计师' },
        { type: 'ai', content: '设计师是个很有创造力的职业！你主要做什么类型的设计？' },
        { type: 'human', content: '我喜欢艺术和音乐' },
        { type: 'ai', content: '艺术和音乐都是很好的爱好，它们能激发创作灵感。' },
        { type: 'human', content: '我擅长 UI/UX 设计' },
        { type: 'ai', content: 'UI/UX 设计非常重要，好的用户体验能让产品更成功！' },
    ];

    for (const msg of messages) {
        if (msg.type === 'human') {
            await history.addMessage(new HumanMessage(msg.content));
        } else {
            await history.addMessage(new AIMessage(msg.content));
        }
    }

    let allMessages = await history.getMessages();

    console.log(`原始消息数量: ${allMessages.length}`);
    console.log(`原始消息: ${allMessages.map(msg => msg.content).join('\n')}`);

    if (allMessages.length > maxMessages) {
        const keepRecent = 2
        const recentMessages = allMessages.slice(-keepRecent);
        const messagesToSummarize = allMessages.slice(0, -keepRecent);

        console.log(`\n历史消息过多，开始总结`)
        console.log(`将被总结的消息数量: ${messagesToSummarize.length}`);
        console.log(`将被保留的消息数量: ${keepRecent}`);

        const summary = await summarizationHistory(messagesToSummarize);

        await history.clear()
        for (const msg of recentMessages) {
            await history.addMessage(msg);
        }

        console.log(`\n 保留的消息数量: ${recentMessages.length}`);
        console.log(`保留的消息: ${recentMessages.map(msg => msg.content).join('\n')}`);
        console.log(`\n 总结的内容: ${summary}`);
    }
}

summarizationMemoryDemo();
