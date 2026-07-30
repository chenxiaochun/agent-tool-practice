import 'dotenv/config';
import { parse } from 'path'
import { MilvusClient, MetricType, DataType, IndexType } from '@zilliz/milvus2-sdk-node';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { EPubLoader } from '@langchain/community/document_loaders/fs/epub';
import { RecursiveCharacterTextSplitter } from '@langchain/classic/text_splitter';

const COLLECTION_NAME = 'ebook_collection';
/** 与 src/model.mjs / naive-rag 保持一致；text-embedding-v3 常用 1024 维 */
const VECTOR_DIM = 1024;
const CHUNK_SIZE = 500;

const EPUB_FILE = '/Users/chenxiaochun/Documents/MyProject/agent-tool-practice/src/milvus/天龙八部.epub';
const BOOK_NAME = parse(EPUB_FILE).name;
/** 与 docker-compose / naive-rag 一致；可用 MILVUS_ADDRESS 覆盖 */
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS ?? 'localhost:19530';

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
    dimensions: VECTOR_DIM,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const client = new MilvusClient({
    address: MILVUS_ADDRESS,
});

/** 限流时退避重试，避免整章 Promise.all 一把梭直接 429 */
async function getEmbeddings(text, retries = 6) {
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            return await embeddings.embedQuery(text);
        } catch (error) {
            const isRateLimit =
                error?.status === 429 ||
                error?.lc_error_code === 'MODEL_RATE_LIMIT' ||
                /rate.?limit|429/i.test(String(error?.message ?? ''));
            if (!isRateLimit || attempt === retries) throw error;
            const waitMs = Math.min(60_000, 2000 * 2 ** attempt);
            console.warn(`embedding 限流，${waitMs}ms 后重试 (${attempt + 1}/${retries})...`);
            await new Promise((r) => setTimeout(r, waitMs));
        }
    }
}

/** 该章首段 id 已存在则视为已入库，支持断点续跑 */
async function chapterAlreadyInserted(bookId, chapterIndex) {
    const firstId = bookId * 100000000 + chapterIndex * 10000;
    const result = await client.query({
        collection_name: COLLECTION_NAME,
        filter: `id == ${firstId}`,
        output_fields: ['id'],
        limit: 1,
    });
    return (result.data?.length ?? 0) > 0;
}

async function enuseCollection(bookId) {
    try {
        const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
        if (!hasCollection.value) {
            console.log('创建集合')
            await client.createCollection({
                collection_name: COLLECTION_NAME,
                fields: [
                    {
                        name: 'id',
                        data_type: DataType.Int64,
                        max_length: 100,
                        is_primary_key: true,
                    },
                    {
                        name: 'book_id',
                        data_type: DataType.Int64,
                        max_length: 100,
                    },
                    {
                        name: 'book_name',
                        data_type: DataType.VarChar,
                        max_length: 100,
                    },
                    {
                        name: 'chapter_name',
                        data_type: DataType.VarChar,
                        max_length: 200,
                    },
                    {
                        name: 'index',
                        data_type: DataType.Int64,
                    },
                    {
                        name: 'content',
                        data_type: DataType.VarChar,
                        max_length: 2000,
                    },
                    {
                        name: 'vector',
                        data_type: DataType.FloatVector,
                        dim: VECTOR_DIM,
                    }
                ]
            })
            console.log('集合创建成功')

            console.log('创建索引...')
            await client.createIndex({
                collection_name: COLLECTION_NAME,
                field_name: 'vector',
                index_type: IndexType.FLAT,
                metric_type: MetricType.COSINE,
            })
            console.log('索引创建成功')
        }

        try {
            await client.loadCollection({
                collection_name: COLLECTION_NAME,
            })
            console.log('集合加载成功')
        } catch (error) {
            console.error('Error loading collection:', error);
            process.exit(1);
        }
    } catch (error) {
        console.error('Error connecting to Milvus:', error);
        process.exit(1);
    }
}

async function insertChunkBatch(chunks, bookId, chapterIndex, chapterName) {
    try {
        if (chunks.length === 0) {
            return 0;
        }

        // 逐条生成向量（比整章并发更稳，少触发限流）
        const insertData = [];
        for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
            const vector = await getEmbeddings(chunks[chunkIndex]);
            insertData.push({
                id: bookId * 100000000 + chapterIndex * 10000 + chunkIndex,
                book_id: bookId,
                book_name: BOOK_NAME,
                chapter_name: chapterName,
                index: chunkIndex,
                content: chunks[chunkIndex],
                vector,
            });
            if ((chunkIndex + 1) % 10 === 0 || chunkIndex + 1 === chunks.length) {
                console.log(`  向量进度 ${chunkIndex + 1}/${chunks.length}`);
            }
        }

        const insertResult = await client.insert({
            collection_name: COLLECTION_NAME,
            data: insertData,
        })

        if (insertResult.status?.error_code !== 'Success') {
            throw new Error(insertResult.status?.reason ?? 'Insert failed');
        }

        const insertedCount = Number(insertResult.insert_cnt) || insertResult.succ_index?.length || insertData.length;
        console.log(`插入完成，共${insertedCount}段`)
        return insertedCount;
    }
    catch (error) {
        console.error('Error inserting chunk batch:', error);
        process.exit(1);
    }
}

async function loadAndProcessEbook(bookId) {
    try {
        console.log(`开始加载电子书${BOOK_NAME}...`)

        const loader = new EPubLoader(EPUB_FILE, { splitChapters: true });
        const documents = await loader.load();
        console.log(`加载完成，共${documents.length}章`)

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: CHUNK_SIZE,
            chunkOverlap: 50,
        });

        let totalInserted = 0;
        for (let i = 0; i < documents.length; i++) {
            const chapter = documents[i];
            const chapterContent = chapter.pageContent;

            console.log(`处理第${i + 1}/${documents.length}章`)
            const chunks = await textSplitter.splitText(chapterContent);
            console.log(`分割完成，共${chunks.length}段`)

            if (chunks.length === 0) {
                console.log(`第${i + 1}章没有内容，跳过`)
                continue;
            }

            if (await chapterAlreadyInserted(bookId, i)) {
                console.log(`第${i + 1}章已存在，跳过（断点续跑）`)
                continue;
            }

            console.log('生成向量...')
            const chapterName = chapter.metadata.title || `第${i + 1}章`;
            const insertedCount = await insertChunkBatch(chunks, bookId, i, chapterName);
            totalInserted += insertedCount;
            console.log(`插入完成，共${insertedCount}段`)

            console.log(`第${i + 1}章插入完成，共${totalInserted}段`)
        }
    }
    catch (error) {
        console.error('Error loading and processing ebook:', error);
        process.exit(1);
    }
}

async function retrieveRelevalantContent(question, k = 3) {
    try {
        const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
        if (!hasCollection.value) {
            console.error(
                `集合 ${COLLECTION_NAME} 不存在。请先取消注释 main 里的 enuseCollection / loadAndProcessEbook 完成入库。`,
            );
            return [];
        }

        const queryVector = await getEmbeddings(question);
        const searchResult = await client.search({
            collection_name: COLLECTION_NAME,
            vector: queryVector,
            limit: k,
            metric_type: MetricType.COSINE,
            output_fields: ['id', 'book_id', 'book_name', 'chapter_name', 'index', 'content'],
        })
        return searchResult.results;
    } catch (error) {
        console.error('Error retrieving relevalant content:', error);
        return [];
    }
}

async function answerQuestion(question, k = 3) {
    try {
        const relevalantContent = await retrieveRelevalantContent(question, k);
        if (relevalantContent.length === 0) {
            return '没有找到相关内容';
        }

        relevalantContent.forEach(item => {
            console.log(`ID：${item.id}`)
            console.log(`Book ID: ${item.book_id}`)
            console.log(`Book Name: ${item.book_name}`)
            console.log(`Chapter Name: ${item.chapter_name}`)
            console.log(`Index: ${item.index}`)
            console.log(`内容: ${item.content}`)
            console.log(`--------------------------------`)
        });

        const context = relevalantContent.map(item => `${item.chapter_name}第${item.index}段: ${item.content}`).join('\n');

        const prompt = `
        你是一个专业的书籍问答助手，请根据以下内容回答问题：
        ${context}
        问题：${question}
        回答要求：
        1. 回答问题时，请使用中文回答
        2. 可以综合多个片段的信息来回答问题
        3. 如果片段中没有相关信息，请如实告知用户
        `

        const response = await model.invoke(prompt)
        return response.content;
    } catch (error) {
        console.error('Error answering question:', error);
        return [];
    }
}

async function main() {
    try {
        console.log(`Connecting to Milvus (${MILVUS_ADDRESS})...`);
        await client.connectPromise;
        console.log('Connected to Milvus');

        const bookId = 1;
        // 首次 / 断点续跑时打开；数据已齐后请保持注释，只做问答
        // await enuseCollection(bookId);
        // await loadAndProcessEbook(bookId);

        // 问答前确保集合已加载（若刚重启过 Milvus）
        await client.loadCollection({ collection_name: COLLECTION_NAME });

        console.log('Searching similar chapters...')
        const query = '乔峰会什么武功？'
        const answer = await answerQuestion(query);
        console.log(`回答：${answer}`)
    }
    catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

main()