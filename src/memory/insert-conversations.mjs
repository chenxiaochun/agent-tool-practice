import 'dotenv/config';
import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';
import { OpenAIEmbeddings } from '@langchain/openai';

const COLLECTION_NAME = 'conversations';
const VECTOR_DIM = 1024;

const embeddingModel = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: process.env.EMBEDDINGS_MODEL_NAME,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const client = new MilvusClient({
    address: '192.168.3.19:19530',
})

async function getEmbeddings(text) {
    const result = await embeddingModel.embedQuery(text);
    return result;
}

async function main() {
    try {
        console.log('Connecting to Milvus...');
        await client.connectPromise
        console.log('Connected to Milvus');

        const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
        if (hasCollection.value) {
            console.log(`集合 "${COLLECTION_NAME}" 已存在，正在删除...`);
            await client.dropCollection({ collection_name: COLLECTION_NAME });
        }

        console.log('创建集合')
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
                    name: 'content', data_type: DataType.VarChar, max_length: 2000, description: 'Conversation content'
                },
                {
                    name: 'round', data_type: DataType.Int64, description: 'Round number'
                },
                {
                    name: 'timestamp', data_type: DataType.Int64, description: 'Timestamp'
                }
            ]
        })
        console.log('集合已创建')

        console.log('创建索引')
        await client.createIndex({
            collection_name: COLLECTION_NAME,
            field_name: 'vector',
            index_type: IndexType.FLAT,
            metric_type: MetricType.COSINE,
        })
        console.log('索引已创建')

        console.log('加载集合')
        await client.loadCollection({
            collection_name: COLLECTION_NAME,
        })
        console.log('集合已加载')

        console.log('插入对话数据')
        const conversations = [
            {
                id: 1,
                content: '我是小明，我今年 8 岁，我喜欢吃苹果',
                round: 1,
                timestamp: Date.now(),
            },
            {
                id: 2,
                content: '我是小红，我今年 10 岁，我喜欢吃香蕉',
                round: 2,
                timestamp: Date.now(),
            },
            {
                id: 3,
                content: '我是小刚，我今年 12 岁，我喜欢吃西瓜',
                round: 3,
                timestamp: Date.now(),
            }
        ]

        console.log('生成向量数据')
        const conversationData = await Promise.all(
            conversations.map(async (item) => {
                const vector = await getEmbeddings(item.content);
                return {
                    ...item,
                    vector
                }
            })
        )

        const insertResult = await client.insert({
            collection_name: COLLECTION_NAME,
            data: conversationData,
        })

        if (insertResult.status?.error_code !== 'Success') {
            throw new Error(insertResult.status?.reason ?? 'Insert failed');
        }

        await client.flush({ collection_names: [COLLECTION_NAME] });

        const insertedCount = Number(insertResult.insert_cnt) || insertResult.succ_index?.length || conversationData.length;
        console.log(`插入 ${insertedCount} 条数据`)
        console.log('='.repeat(80))

    } catch (error) {
        console.error('Error connecting to Milvus:', error);
        process.exit(1);
    }
}

main()
