import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import { StructuredOutputParser } from '@langchain/core/output_parsers';
import {
  initLangfuseTracing,
  shutdownLangfuseTracing,
  traceLangChainRun,
} from './langfuse/tracing.mjs';

const model = new ChatOpenAI({
  modelName: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  temperature: 0,
});

const schema = z.object({
  name: z.string().describe('姓名'),
  gender: z.string().describe('性别'),
  birthdate: z.string().describe('出生日期'),
  nationality: z.string().describe('国籍'),
  achievements: z.string().describe('主要成就'),
  influence: z.string().describe('影响力'),
});

const parser = StructuredOutputParser.fromZodSchema(schema);
const prompt = `详细介绍莫扎特的信息。${parser.getFormatInstructions()}`;

initLangfuseTracing();

try {
  const stream = await model.stream(prompt);
  let chunkTotal = '';
  let chunkCount = 0;
  for await (const chunk of stream) {
    chunkTotal += chunk.content;
    chunkCount++;
    process.stdout.write(chunk.content);
  }
  console.log(`流式输出完成，共 ${chunkCount} 个 chunk`);

  const result = await parser.parse(chunkTotal);
  console.log(`姓名：${result.name}`);
  console.log(`性别：${result.gender}`);
  console.log(`出生日期：${result.birthdate}`);
  console.log(`国籍：${result.nationality}`);
  console.log(`主要成就：${result.achievements}`);
  console.log(`影响力：${result.influence}`);
} catch (error) {
  console.error('Error generating answer:', error);
}
