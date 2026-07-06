import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
})

const scienceSchema = z.object({
    name: z.string().describe('姓名'),
    gender: z.string().describe('性别'),
    birthdate: z.string().describe('出生日期'),
    nationality: z.string().describe('国籍'),
    achievements: z.string().describe('主要成就'),
    influence: z.string().describe('影响力'),
})


const question = `请介绍一下爱因斯坦的信息`
const structuredModel = model.withStructuredOutput(scienceSchema)

const response = await structuredModel.invoke(question)
console.log(JSON.stringify(response, null, 2))