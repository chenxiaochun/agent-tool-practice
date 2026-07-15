import 'dotenv/config'
import { RunnableWithMessageHistory } from '@langchain/core/runnables'
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history'
import { ChatOpenAI } from '@langchain/openai'
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts'

import { StringOutputParser } from '@langchain/core/output_parsers'

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
})

const prompt = ChatPromptTemplate.fromMessages([
    ['system', '你是一个助手，请根据用户的问题给出回答。'],
    new MessagesPlaceholder('history'),
    ['human', '{question}'],
])

const simpleChain = prompt.pipe(model).pipe(new StringOutputParser())

const messageHistories = new Map()

const getMessageHistory = (sessionId) => {
    if (!messageHistories.has(sessionId)) {
        messageHistories.set(sessionId, new InMemoryChatMessageHistory())
    }
    return messageHistories.get(sessionId)
}

const chain = new RunnableWithMessageHistory({
    runnable: simpleChain,
    getMessageHistory,
    inputMessagesKey: 'question',
    historyMessagesKey: 'history',
})

console.log('第一次对话：')
const result1 = await chain.invoke(
    {
        question: '你好，我是小明，我喜欢编程，我住在北京',
    },
    {
        configurable: {
            sessionId: '123',
        },
    }
)
console.log(result1)

console.log('\n第二次对话：')
const result2 = await chain.invoke(
    {
        question: '我叫什么名字？我喜欢什么？我住在哪里？',
    },
    {
        configurable: {
            sessionId: '123',
        },
    }
)
console.log(result2)
