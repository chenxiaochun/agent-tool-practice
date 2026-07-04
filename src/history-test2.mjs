import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { FileSystemChatMessageHistory } from '@langchain/community/stores/message/file_system';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import path from 'node:path';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

async function fileSystemDemo() {
    const filePath = path.join(process.cwd(), 'chat_history.json');
    const sessionId = 'user_session01';

    const systemMessage = new SystemMessage('你是一个友好，幽默的做菜助手，喜欢分享美食和烹饪技巧');
    console.log('第一次对话：');
    const history = new FileSystemChatMessageHistory({
        filePath,
        sessionId,
    })

    const userMessage1 = new HumanMessage('红烧肉怎么做？');
    await history.addMessage(userMessage1);
    const message1 = [systemMessage, ...(await history.getMessages())];
    const response1 = await model.invoke(message1);
    await history.addMessage(response1);

    console.log(`用户：${userMessage1.content}`)
    console.log(`助手：${response1.content}`)
    console.log(`文件已保存到${filePath}`)

    console.log('第二次对话：');
    const userMessage2 = new HumanMessage('好吃吗？');
    await history.addMessage(userMessage2);
    const message2 = [systemMessage, ...(await history.getMessages())];
    const response2 = await model.invoke(message2);
    await history.addMessage(response2);

    console.log(`用户：${userMessage2.content}`)
    console.log(`助手：${response2.content}`)
    console.log(`文件已保存到${filePath}`)

    console.log('历史消息记录')
    const allMessages = await history.getMessages();
    console.log(`共保存了${allMessages.length}条消息`)
    allMessages.forEach((message, index) => {
        const type = message.type === 'human' ? '用户' : '助手';
        console.log(`${index + 1}. ${type}: ${message.content.substring(0, 100)}...`)
    })
}

fileSystemDemo();