import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import { JsonOutputToolsParser } from '@langchain/core/output_parsers/openai_tools';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_APIKEY,
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

const parser = new JsonOutputToolsParser(schema)
const chain = modelWithTools.pipe(parser)

try {
    const stream = await chain.stream('请介绍一下爱因斯坦的信息')
    const lastContent = ''
    const finalContent = ''

    for await (const chunk of stream) {
        if (chunk.length > 0) {
            const toolCall = chunk[0]
            console.log(toolCall.args)
        }
    }
} catch (error) {
    console.error('Error generating answer:', error);
}

