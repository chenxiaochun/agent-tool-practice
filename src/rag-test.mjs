import 'dotenv/config';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory';


const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
});

const embeddings = new OpenAIEmbeddings({
    modelName: process.env.EMBEDDINGS_MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const documents = [
    new Document({
        pageContent: '小明是一个小学生，他喜欢打篮球，喜欢看漫画，喜欢玩电脑游戏。',
        metadata: {
            chapter: '1',
            character: '小明',
            type: '角色介绍',
            mod: '活泼'
        },
    }),
    new Document({
        pageContent: '小红是一个小学生，她喜欢唱歌，喜欢跳舞，喜欢画画。',
        metadata: {
            chapter: '1',
            character: '小红',
            type: '角色介绍',
            mod: '活泼'
        },
    }),
    new Document({
        pageContent: '有一天，学校要举办文艺汇演，小明和小红都报名了。',
        metadata: {
            chapter: '2',
            character: '小明',
            type: '事件描述',
            mod: '活泼'
        },
    }),
    new Document({
        pageContent: '后来，小明和小红都获得了文艺汇演的冠军。',
        metadata: {
            chapter: '2',
            character: '小明',
            type: '事件描述',
            mod: '活泼'
        },
    })
]

const vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);
const retriever = vectorStore.asRetriever({ k: 1 });

const questions = [
    '小明和小红怎么获得了什么奖项？'
];

for (const question of questions) {
    console.log('='.repeat(80));
    console.log(`🔍 问题: ${question}`);
    console.log('='.repeat(80));

    const retrievedDocs = await retriever.invoke(question);
    const scoredResults = await vectorStore.similaritySearchWithScore(question, 3);

    retrievedDocs.forEach((doc, i) => {
        const scoredResult = scoredResults.find(([scoredDoc]) => scoredDoc.pageContent === doc.pageContent);
        const score = scoredResult ? scoredResult[1] : null;
        const similarity = score ? (1 - score).toFixed(4) : 'N/A';

        console.log(`🔍 相关文档 ${i + 1}: ${doc.pageContent}`);
        console.log(`🔍 相似度: ${similarity}`);
        console.log('='.repeat(80));
    })

    const context = retrievedDocs.map((doc, i) => `相关文档 ${i + 1}: ${doc.pageContent}`).join('\n');
    const prompt = `你是一个讲友情故事的作家，请根据以下内容创作一个友情故事：
        ${context}

        问题：${question}

        友情故事：`;

    const response = await model.invoke(prompt);
    console.log(response.content);
}
