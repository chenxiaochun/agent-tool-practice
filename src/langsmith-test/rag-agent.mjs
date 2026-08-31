import 'dotenv/config';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';
import { Milvus } from '@langchain/community/vectorstores/milvus';
import { traceLangChainRun } from '../langfuse/tracing.mjs';
import { embeddings, model } from '../model.mjs';

const vectorStore = await Milvus.fromExistingCollection(embeddings, {
  collectionName: process.env.MILVUS_COLLECTION ?? 'rag_docs',
  url: process.env.MILVUS_URI ?? 'http://localhost:19530',
});

const retriever = vectorStore.asRetriever({ k: 4 });

const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    '你是客服助手。仅根据下面「上下文」回答；上下文没有的信息请明确说不知道，不要编造。\n\n上下文：\n{context}',
  ],
  ['human', '{question}'],
]);

const chain = RunnableSequence.from([prompt, model, new StringOutputParser()]);

const GraphState = Annotation.Root({
  question: Annotation,
  context: Annotation,
  answer: Annotation,
});

async function retrieve(state) {
  const docs = await retriever.invoke(state.question);
  return { context: docs };
}

async function generate(state) {
  const contextText = state.context.map((d) => d.pageContent).join('\n\n');
  const answer = await chain.invoke({
    context: contextText,
    question: state.question,
  });
  return { answer };
}

const workflow = new StateGraph(GraphState)
  .addNode('retrieve', retrieve)
  .addNode('generate', generate)
  .addEdge(START, 'retrieve')
  .addEdge('retrieve', 'generate')
  .addEdge('generate', END);

export const ragApp = workflow.compile();

export async function ask(question, options = {}) {
  const { output, traceId } = await traceLangChainRun(
    'customer-support-rag',
    { question },
    async (langfuseHandler) => {
      const result = await ragApp.invoke(
        { question },
        {
          callbacks: [langfuseHandler],
          runName: 'customer-support-rag',
          metadata: {
            feature: 'customer-support',
            ...(options.sessionId ? { langfuseSessionId: options.sessionId } : {}),
            ...(options.userId ? { langfuseUserId: options.userId } : {}),
          },
        },
      );

      return {
        answer: result.answer,
        context: result.context ?? [],
      };
    },
    {
      sessionId: options.sessionId,
      userId: options.userId,
      tags: ['rag', 'customer-support', ...(options.tags ?? [])],
    },
  );

  return {
    ...output,
    traceId,
  };
}
