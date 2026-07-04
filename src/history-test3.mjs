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

async function fileHistoryDemo() {
    const filePath = path.join(process.cwd(), 'chat_history.json');
    const sessionId = 'user_session01';

    const systemMessage = new SystemMessage('你是一个友好，幽默的做菜助手，喜欢分享美食和烹饪技巧');
    const restoredHistory = new FileSystemChatMessageHistory({
        filePath: filePath,
        sessionId: sessionId,
    })
    const restoredMessages = await restoredHistory.getMessages();
    console.log(`从文件中恢复了${restoredMessages.length}条消息`);
    restoredMessages.forEach((msg, index) => {
        const type = msg.type === 'human' ? '用户' : '助手';
        console.log(`${index + 1}. ${type}: ${msg.content.substring(0, 100)}...`)
    })

    console.log('第三次对话：');
    const userMessage3 = new HumanMessage(' 需要哪些食材？');
    await restoredHistory.addMessage(userMessage3);
    const message3 = [systemMessage, ...(await restoredHistory.getMessages())];
    const response3 = await model.invoke(message3);
    await restoredHistory.addMessage(response3);

    console.log(`用户：${userMessage3.content}`)
    console.log(`助手：${response3.content}`)
    console.log(`文件已保存到${filePath}`)
}

fileHistoryDemo();