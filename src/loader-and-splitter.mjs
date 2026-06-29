import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import 'cheerio'
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from '@langchain/classic/text_splitter';
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory';
import { OpenAIEmbeddings } from '@langchain/openai';

const cheerioLoader = new CheerioWebBaseLoader('https://juejin.cn/post/7233327509919547452', {
    selector: '.main-area p',
});

const documents = await cheerioLoader.load();
// console.log(documents)

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 400,
    chunkOverlap: 50,
    separators: ['。', '！', '？'],
});

const splitDocuments = await textSplitter.splitDocuments(documents);
console.log(splitDocuments)

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const embeddings = new OpenAIEmbeddings({
    modelName: process.env.EMBEDDINGS_MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

console.log('🔍 向量化开始...');
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocuments, embeddings);
console.log('🔍 向量化完成');

const retriever = vectorStore.asRetriever({ k: 3 });
console.log('🔍 检索器创建完成');

const questions = [
    '父亲的去世对创作者产生了怎样的影响？'
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
    const prompt = `你是一个专业的文章分析专家，请根据以下内容回答问题：
        ${context}

        问题：${question}

        回答：`;

    const response = await model.invoke(prompt);
    console.log(response.content);
}