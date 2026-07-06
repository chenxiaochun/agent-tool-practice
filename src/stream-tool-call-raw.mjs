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

const schema = z.object({
    name: z.string().describe('姓名'),
    gender: z.string().describe('性别'),
    birthdate: z.string().describe('出生日期'),
    nationality: z.string().describe('国籍'),
    achievements: z.string().describe('主要成就'),
    influence: z.string().describe('影响力'),
})

const modelWithTools = model.bindTools([
    {
        name: 'get_science_info',
        description: '获取科学信息',
        schema: schema,
    }
])

const query = '请介绍一下爱因斯坦的信息'

const stream = await modelWithTools.stream(query)

let chunkIndex = 0
for await (const chunk of stream) {
    chunkIndex++
    if (chunk.tool_call_chunks?.length > 0) {
        const args = chunk.tool_call_chunks[0].args
        if (args) {
            process.stdout.write(args)
        }
    }
}