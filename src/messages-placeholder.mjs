import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
})

const chatPromptWithHistory = ChatPromptTemplate.fromMessages([
    [
        'system',
        '你是一名资深工程效率顾问，擅长分析团队开发效率问题，并提出改进建议。'
    ],
    new MessagesPlaceholder('history'),
    [
        'human',
        '这是本轮用户的新问题：{current_question}，请结合历史对话内容，给出回答。'
    ]
])

const historyMessages = [
    {
        role: 'human',
        content: '我们团队最近在做一个内部的周报自动生成工具。',
    },
    {
        role: 'ai',
        content: '听起来不错，可以先把数据源（Git / Jira / 运维）梳理清楚，再考虑 Prompt 模块化设计。',
    },
    {
        role: 'human',
        content: '我们已经把 Prompt 拆成了「人设」「背景」「任务」「格式」四块。',
    },
    {
        role: 'ai',
        content: '很好，接下来可以考虑把这些模块做成可复用的 PipelinePromptTemplate，方便在不同场景复用。',
    },
];

async function main() {
    const formattedPrompt = await chatPromptWithHistory.format({
        history: historyMessages,
        current_question: '我们团队最近在做一个内部的周报自动生成工具。',
    })
    console.log(formattedPrompt)

    const response = await model.invoke(formattedPrompt)
    console.log(response.content)
}

main()