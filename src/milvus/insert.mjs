import 'dotenv/config';
import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';
import { OpenAIEmbeddings } from '@langchain/openai';

const COLLECTION_NAME = 'ai_diary';
const VECTOR_DIM = 1536; // text-embedding-v2 默认输出维度

const embeddingModel = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: process.env.EMBEDDINGS_MODEL_NAME,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const client = new MilvusClient({
    address: '192.168.3.2:19530',
})

async function getEmbeddings(text) {
    return embeddingModel.embedQuery(text);
}

async function insertData() {
    try {
        console.log('Connecting to Milvus...');
        await client.connectPromise
        console.log('Connected to Milvus');

        const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
        if (hasCollection.value) {
            console.log(`Collection "${COLLECTION_NAME}" already exists, dropping...`);
            await client.dropCollection({ collection_name: COLLECTION_NAME });
        }

        await client.createCollection({
            collection_name: COLLECTION_NAME,
            fields: [
                {
                    name: 'id', data_type: DataType.Int64, is_primary_key: true, description: 'Primary key'
                },
                {
                    name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM, description: 'Embedding vector'
                },
                {
                    name: 'content', data_type: DataType.VarChar, max_length: 2000, description: 'Diary content'
                },
                {
                    name: 'date', data_type: DataType.VarChar, max_length: 100, description: 'Date'
                },
                {
                    name: 'mood', data_type: DataType.Int64, description: 'Mood'
                },
                {
                    name: 'tags', data_type: DataType.VarChar, max_length: 1000, description: 'Tags'
                },
            ]
        })
        console.log('Collection created successfully');

        console.log('Creating index...');
        await client.createIndex({
            collection_name: COLLECTION_NAME,
            field_name: 'vector',
            index_type: IndexType.FLAT,
            metric_type: MetricType.COSINE,
        })
        console.log('Index created successfully');

        await client.loadCollection({
            collection_name: COLLECTION_NAME,
        })
        console.log('Collection loaded successfully');

        const diary_contents = [
            {
                id: 1,
                content: '今天天气很好，去公园散步了，心情非常愉快,看到了很多美丽的风景，心情非常愉快',
                date: '2026-06-30',
                mood: 1,
                tags: ['天气', '公园', '心情愉快'],
            },
            {
                id: 2,
                content: '今天工作很忙，心情非常糟糕，感觉很累，没有时间休息，心情非常糟糕',
                date: '2026-06-29',
                mood: 0,
                tags: ['工作', '累', '心情糟糕'],
            },
            {
                id: 3,
                content: '今天去爬山了，心情非常愉快，看到了很多美丽的风景，心情非常愉快',
                date: '2026-06-28',
                mood: 0,
                tags: ['爬山', '风景', '心情愉快'],
            }
        ]

        console.log('Generating embeddings...');
        const diary_data = await Promise.all(diary_contents.map(async (diary) => {
            const vector = await getEmbeddings(diary.content);
            return {
                id: diary.id,
                content: diary.content,
                date: diary.date,
                mood: diary.mood,
                tags: diary.tags.join(','),
                vector,
            }
        }));
        console.log('Embeddings generated successfully');

        const insertResult = await client.insert({
            collection_name: COLLECTION_NAME,
            data: diary_data,
        })
        if (insertResult.status?.error_code !== 'Success' && Number(insertResult.insert_cnt) === 0) {
            throw new Error(insertResult.status?.reason ?? 'Insert failed');
        }
        console.log('Data inserted successfully');

    } catch (error) {
        console.error('Error connecting to Milvus:', error);
        process.exit(1);
    }
}

async function queryData() {
    console.log('Connecting to Milvus...');
    await client.connectPromise
    console.log('Connected to Milvus');

    console.log('Searching similar diaries...');
    const query = '我想看关于心情糟糕的日记'
    const queryVector = await getEmbeddings(query);
    const searchResult = await client.search({
        collection_name: COLLECTION_NAME,
        vector: queryVector,
        limit: 2,
        metric_type: MetricType.COSINE,
        output_fields: ['id', 'content', 'date', 'mood', 'tags'],
    })

    searchResult.results.forEach((result) => {
        console.log(`Diary ID: ${result.id}`);
        console.log(`Content: ${result.content}`);
        console.log(`Date: ${result.date}`);
        console.log(`Mood: ${result.mood}`);
    })
}

// insertData();
queryData();