import 'dotenv/config';
import { MilvusClient, MetricType } from '@zilliz/milvus2-sdk-node';
import {
    ChatOpenAI, OpenAIEmbeddings
} from '@langchain/openai';

const COLLECTION_NAME = 'ai_diary';
const VECTOR_DIM = 1536;

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0.7,
})

const embeddings = new OpenAIEmbeddings({
    modelName: process.env.EMBEDDINGS_MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
})

const client = new MilvusClient({
    address: '192.168.3.2:19530',
})

async function getEmbeddings(text) {
    return embeddings.embedQuery(text);
}

async function retrieveRelevantDiaries(query, k = 2) {
    try {
        const queryVector = await getEmbeddings(query);
        const searchResult = await client.search({
            collection_name: COLLECTION_NAME,
            vector: queryVector,
            limit: k,
            metric_type: MetricType.COSINE,
            output_fields: ['id', 'content', 'date', 'mood', 'tags'],
        })
        return searchResult.results
    } catch (error) {
        console.error('Error retrieving relevant diaries:', error);
        process.exit(1);
    }
}

async function answerDiaryQuestion(query) {
    try {
        console.log('='.repeat(80));
        console.log(`🔍 问题: ${query}`);
        console.log('='.repeat(80));

        console.log('🔍 检索相关日记...');
        const relevantDiaries = await retrieveRelevantDiaries(query);
        if (relevantDiaries.length === 0) {
            console.log('🔍 没有检索到相关日记');
            return;
        }

        console.log(`🔍 检索到 ${relevantDiaries.length} 篇相关日记`);
        console.log('='.repeat(80));

        relevantDiaries.forEach((diary) => {
            console.log(`🔍 相关日记 ${diary.id}: ${diary.content}`);
            console.log(`🔍 日期: ${diary.date}`);
            console.log(`🔍 心情: ${diary.mood}`);
            console.log(`🔍 标签: ${diary.tags}`);
            console.log(`🔍 内容: ${diary.content}`);
            console.log('='.repeat(80));
        })

        const context = relevantDiaries.map((diary) => {
            return `
            ID: ${diary.id}
            日期: ${diary.date}
            心情: ${diary.mood}
            标签: ${diary.tags}
            内容: ${diary.content}
            `
        }).join('\n');

        const prompt = `你是一个日记分析师，请根据以下内容回答问题：
            ${context}
            `;

        const response = await model.invoke(prompt);
        console.log(`🔍 AI助手回答: ${response.content}`);
        console.log('='.repeat(80));
        console.log(response.content);
    }
    catch (error) {
        console.error('Error answering diary question:', error);
        process.exit(1);
    }
}

async function main() {
    try {
        console.log('🔍 连接到 Milvus...');
        await client.connectPromise;
        console.log('🔍 连接成功');

        console.log('🔍 开始回答日记问题...');
        const query = '我想看关于心情糟糕的日记';
        await answerDiaryQuestion(query);
    }
    catch (error) {
        console.error('Error in main:', error);
        process.exit(1);
    }
}

main()