import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { AIMessage, HumanMessage, trimMessages } from '@langchain/core/messages';
import { getEncoding } from 'js-tiktoken';

async function messageCountTruncation() {
    const history = new InMemoryChatMessageHistory();
    const maxMessages = 4

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
    const trimmedMessages = allMessages.slice(-maxMessages);
    console.log('截断前的消息数量:', allMessages.length);
    console.log('截断后的消息数量:', trimmedMessages.length);
    console.log('截断后的消息:', trimmedMessages.map(msg => msg.content).join('\n'));
}

function countTokens(message, encoder) {
    let total = 0
    for (const msg of message) {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
        total += encoder.encode(content).length;
    }
    return total;
}

async function tokenCountTruncation() {
    const history = new InMemoryChatMessageHistory();
    const maxTokens = 100
    const enc = getEncoding('cl100k_base');

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

    let allMessages = await history.getMessages()

    const trimmedMessages = await trimMessages(allMessages, {
        maxTokens,
        tokenCounter: async msgs => countTokens(msgs, enc),
        strategy: 'last',
    })

    const totalTokens = await countTokens(allMessages, enc);
    console.log(`总Token数: ${totalTokens}`);
    console.log(`保留消息数量: ${trimmedMessages.length}`);
    console.log(`保留的消息：${trimmedMessages.map(map => {
        const content = typeof map.content === 'string' ? map.content : JSON.stringify(map.content);
        const tokens = enc.encode(content).length;
        return `${map.type}: ${content} (Tokens: ${tokens})`;
    }).join('\n')
        } `)
}

messageCountTruncation();
tokenCountTruncation();