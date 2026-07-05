import "dotenv/config";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { MilvusClient, MetricType } from "@zilliz/milvus2-sdk-node";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const COLLECTION_NAME = 'conversations';
const VECTOR_DIM = 1024;

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const embeddingModel = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: process.env.EMBEDDINGS_MODEL_NAME,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const client = new MilvusClient({
    address: 'localhost:19530',
})

async function getEmbeddings(text) {
    return embeddingModel.embedQuery(text);
}

async function retrieveRelevantConversations(query, k = 2) {
    try {
        const queryVector = await getEmbeddings(query);
        const searchResult = await client.search({
            collection_name: COLLECTION_NAME,
            vector: queryVector,
            limit: k,
            metric_type: MetricType.COSINE,
            output_fields: ['id', 'content', 'round', 'timestamp'],
        })
        return searchResult.results;
    } catch (error) {
        console.error('Error retrieving relevant conversations:', error);
        return [];
    }
}

async function retrivealMemory() {
    try {
        console.log('连接到 Milvus...');
        await client.connectPromise;
        console.log('连接到 Milvus 成功');

        const history = new InMemoryChatMessageHistory();
        const conversations = [
            {
                input: '小明今年几岁？',
            },
            {
                input: '小红今年几岁？',
            },
            {
                input: '小刚喜欢吃什么？',
            }
        ]

        for (let i = 0; i < conversations.length; i++) {
            const { input } = conversations[i]
            const userMessage = new HumanMessage(input);

            console.log(`第 ${i + 1} 轮对话: ${input}`);

            console.log('检索相关历史对话')
            const retrievedConversations = await retrieveRelevantConversations(input);

            let relevantHistory = ''
            if (retrievedConversations.length > 0) {
                retrievedConversations.forEach((conv, index) => {
                    console.log(`历史对话${index + 1} 相似度：${conv.score.toFixed(4)}`)
                    console.log(`轮次：${conv.round}`)
                    console.log(`内容：${conv.content}`)
                })

                relevantHistory = retrievedConversations.map((conv, index) => {
                    return `[历史对话${index + 1}] 轮次：${conv.round} 内容：${conv.content}`
                }).join('\n======\n')
            } else {
                console.log('未找到相关历史对话')
            }

            const contextMessages = relevantHistory ? [new HumanMessage(`相关历史问题：\n${relevantHistory}，用户问题：${input}`)] : [userMessage]
            console.log('AI 回复：')
            const response = await model.invoke(contextMessages)

            history.addMessage(userMessage)
            history.addMessage(response)

            const conversationText = `
            用户问题：${input}
            AI 回复：${response.content}
        `
            const convId = `conv_${Date.now()}_${i + 1}`
            const convVector = await getEmbeddings(conversationText)

            try {
                client.insert({
                    collection_name: COLLECTION_NAME,
                    data: [
                        {
                            id: convId,
                            vector: convVector,
                            content: conversationText,
                            round: i + 1,
                            timestamp: Date.now().toString(),
                        }
                    ]
                })
                console.log('已保存到 Milvus 成功')
            } catch (error) {
                console.error('Error inserting conversation into Milvus:', error);
            }
        }
    } catch (error) {
        console.error('Error connecting to Milvus:', error);
        return [];
    }
}

retrivealMemory();