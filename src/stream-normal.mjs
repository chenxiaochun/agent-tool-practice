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

console.log('普通流式输出：')

const query = '请介绍一下爱因斯坦的信息'

const schema = z.object({
    name: z.string().describe('姓名'),
    gender: z.string().describe('性别'),
    birthdate: z.string().describe('出生日期'),
    nationality: z.string().describe('国籍'),
    achievements: z.string().describe('主要成就'),
    influence: z.string().describe('影响力'),
})

const structuredModel = model.withStructuredOutput(schema)
const stream = await structuredModel.stream(query)

let chunkCount = 0
let result = ''

for await (const chunk of stream) {
    result = chunk
    chunkCount++
    console.log(`流式输出完成，共 ${chunkCount} 个 chunk`)
}

if (result) {
    console.log('最终结果：', result)
    console.log(`姓名：${result.name}`)
    console.log(`性别：${result.gender}`)
    console.log(`出生日期：${result.birthdate}`)
    console.log(`国籍：${result.nationality}`)
    console.log(`主要成就：${result.achievements}`)
    console.log(`影响力：${result.influence}`)
}