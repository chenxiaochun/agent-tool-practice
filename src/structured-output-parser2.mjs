import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { JsonOutputParser, StructuredOutputParser } from '@langchain/core/output_parsers';
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

const parser = StructuredOutputParser.fromZodSchema(scienceSchema)

const question = `请介绍一下爱因斯坦的信息${parser.getFormatInstructions()}`
console.log(question)

try {
    console.log('开始生成回答...')
    const response = await model.invoke(question)
    console.log('回答生成完成')

    const jsonResult = await parser.parse(response.content)
    console.log(jsonResult)

} catch (error) {
    console.error('Error generating answer:', error);
}