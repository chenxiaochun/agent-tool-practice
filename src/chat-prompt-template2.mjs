import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from '@langchain/core/prompts';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
})

const systemMessage = SystemMessagePromptTemplate.fromTemplate(`
    你是一位资深工程团队负责人，写作风格：{tone}    
   你擅长把复杂的技术问题转化为通俗易懂的文字，让非技术人员也能理解。
`)

const humanMessage = HumanMessagePromptTemplate.fromTemplate(`
    以下是本周团队的开发活动
    {dev_activities}
    请你从这些活动中提炼出：
    1. 本周团队的主要亮点
    2. 潜在风格和技术债
    3. 下周计划
`)

const composedPrompt = ChatPromptTemplate.fromMessages([
    systemMessage,
    humanMessage,
])

async function main() {
    const chatMessages = await composedPrompt.format({
        tone: '专业但有人情味',
        dev_activities: '完成字节跳动技术团队的目标,完成字节跳动技术团队的目标,完成字节跳动技术团队的目标',
    })
    console.log(chatMessages)

    const response = await model.invoke(chatMessages)
    console.log(response.content)
}

main()