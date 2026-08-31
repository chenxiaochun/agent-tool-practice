import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import {
  initLangfuseTracing,
  shutdownLangfuseTracing,
  traceLangChainRun,
} from './tracing.mjs';

initLangfuseTracing();

const model = new ChatOpenAI({
  modelName: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

const question = process.argv.slice(2).join(' ') || '用一句话介绍 Langfuse 是什么';

const { output, traceId } = await traceLangChainRun(
  'hello-langchain',
  { question },
  async (langfuseHandler) => {
    const response = await model.invoke(question, {
      callbacks: [langfuseHandler],
      runName: 'hello-langchain',
    });

    return { answer: response.content };
  },
  {
    sessionId: 'hello-demo-session',
    tags: ['hello-langchain', 'demo'],
  },
);

console.log(`问: ${question}`);
console.log(`答: ${output.answer}`);
console.log(`Langfuse trace: ${traceId ?? 'unknown'}`);

await shutdownLangfuseTracing();
