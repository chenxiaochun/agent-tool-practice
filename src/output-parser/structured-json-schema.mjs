import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import chalk from 'chalk';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { zodToJsonSchema } from 'zod-to-json-schema';

const scientistSchema = z.object({
    name: z.string().describe('科学家姓名'),
    birth_year: z.number().describe('科学家出生年份'),
    field: z.string().describe('科学家领域'),
    achievements: z.array(z.string()).describe('科学家成就'),
}).strict()

const nativeJsonSchema = zodToJsonSchema(scientistSchema)

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
    modelKwargs: {
        response_format: {
            type: 'json_schema',
            json_schema: {
                name: 'scientist_info',
                strict: true,
                schema: nativeJsonSchema,
            },
        },
    },
});

async function testNativeJsonSchema() {
    console.log(chalk.bgGreen('🔍 正在测试原生 JSON Schema...'));
    const response = await model.invoke([
        new SystemMessage('你是一个信息提取助手， 请直接返回 JSON 数据'),
        new HumanMessage('请介绍一下杨振宁'),
    ])

    console.log(chalk.bgGreen('🔍 提取结果:'));
    console.log(response.content);
}

testNativeJsonSchema().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});