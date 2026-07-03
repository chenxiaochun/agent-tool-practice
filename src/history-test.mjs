import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

async function inMemoryDemo() {
    const history = new InMemoryChatMessageHistory();
    const systemMessage = new SystemMessage('你是一个友好，幽默的做菜助手，喜欢分享美食和烹饪技巧');

    console.log('第一次对话：');
    const userMessage1 = new HumanMessage('你今天吃的什么？');
    await history.addMessage(userMessage1);

    const message1 = [systemMessage, ...(await history.getMessages())];
    const response1 = await model.invoke(message1);
    await history.addMessage(response1);

    console.log(`用户：${userMessage1.content}`)
    console.log(`助手：${response1.content}`)

    console.log('第二次对话：');
    const userMessage2 = new HumanMessage('好吃吗？');
    await history.addMessage(userMessage2);

    const message2 = [systemMessage, ...(await history.getMessages())];
    const response2 = await model.invoke(message2);
    await history.addMessage(response2);

    console.log(`用户：${userMessage2.content}`)
    console.log(`助手：${response2.content}`)

    console.log('历史消息记录')
    const allMessages = await history.getMessages();
    console.log(`共保存了${allMessages.length}条消息`)
    allMessages.forEach((message, index) => {
        const type = message.type === 'human' ? '用户' : '助手';
        console.log(`${index + 1}. ${type}: ${message.content.substring(0, 100)}...`)
    })
}

inMemoryDemo();