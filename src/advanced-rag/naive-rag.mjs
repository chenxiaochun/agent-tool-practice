/**
 * ============================================================================
 * 新手导读：这份文件在干什么？
 * ============================================================================
 *
 * 目标：做一个最朴素的 RAG（Retrieval-Augmented Generation，检索增强生成）演示：
 * 先从向量库里「翻出」和问题相关的小说片段，再把片段塞进提示词，让大模型据此回答。
 *
 * 类比：
 * - 向量库（Milvus）= 图书馆里按「语义相近」排架的书架（不是按书名拼音，而是意思像不像）
 * - embeddings  = 把文字翻译成「坐标」的尺子；问题和书页都量成坐标，才能比远近
 * - retrieve    = 图书管理员按问题去书架上抽 TOP_K 本最相关的页
 * - generate    = 你把抽到的页摊在桌上，请专家（大模型）对着页回答
 * - LangGraph   = 把「检索 → 生成」两站串成一条流水线（状态图）
 *
 * 为什么叫 Naive（朴素）RAG？
 * - 只做「检索 + 生成」两步，没有：重排序、查询改写、多跳检索、拒答路由等进阶技巧
 * - 适合先把主链路跑通，再对照 advanced-rag 里更复杂的变体
 *
 * 数据前提：Milvus 里已有名为 ebook_collection 的集合（小说片段 + 向量已入库）。
 * 本文件只「连已有集合」，不会新建或写入文档。
 *
 * ----------------------------------------------------------------------------
 * 主流程怎么走？（对应下面的 main + LangGraph）
 * ----------------------------------------------------------------------------
 *  1. 定义图状态 GraphState（问题、k、检索结果、生成答案）
 *  2. 写 retrieveNode / generateNode，并用边串成 START → retrieve → generate → END
 *  3. main 里连上 Milvus（fromExistingCollection）并 loadCollection
 *  4. graph.invoke 塞入 question / k，跑完整条图
 *  5. 打印检索到的片段；生成节点里已用 stream 边出边印答案
 *
 * 新手易混点：
 * - 检索结果不会自动进模型！必须在 generateNode 里拼进 prompt（本文件就是这么做的）
 * - similaritySearchWithScore 返回的 score 含义取决于度量（这里 COSINE）；别和「距离越小越好」混用
 *
 * 阅读建议：先看 GraphState → retrieveNode → generateNode → main 里的 invoke。
 * ============================================================================
 */

import "dotenv/config";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { Milvus } from "@langchain/community/vectorstores/milvus";
/** model = 聊天大模型；embeddings = 把文本变成向量的模型（见 ../model.mjs） */
import { model, embeddings } from '../model.mjs'


// ============================================================================
// 配置：连哪张「书架」、每次最多抽几页
// ============================================================================

/** Milvus 集合名：里面应已存好《天龙八部》等片段的向量与元数据 */
const COLLECTION_NAME = "ebook_collection";
/** 默认检索条数：越大上下文越全，也越费 token、越容易夹杂噪声 */
const TOP_K = 5;

// ============================================================================
// 1. 图状态：整条 RAG 流水线共用的「草稿纸」
// ============================================================================

/**
 * GraphState 四个字段在图里怎么传：
 * - question   ：用户问题（retrieve / generate 都要用）
 * - k          ：检索条数（可覆盖 TOP_K）
 * - documents  ：retrieve 写出 → generate 读入（真正的「检索结果」）
 * - generation ：generate 写出最终回答文本
 *
 * 这里用 Annotation 简写（未自定义 reducer），多次更新同字段时以「后写覆盖」为主。
 */
const GraphState = Annotation.Root({
    question: Annotation,
    k: Annotation,
    documents: Annotation,
    generation: Annotation,
});

/**
 * vectorStore：Milvus 向量库客户端。
 * 故意放在模块级、在 main 里赋值——节点函数闭包能读到它；
 * 必须先连库再 invoke，否则 retrieve 会空引用。
 */
let vectorStore;

// ============================================================================
// 2. 检索助手：把「问题」变成「带元数据的片段列表」
// ============================================================================

/**
 * 向向量库做相似度搜索，并整理成后面拼 prompt / 打印日志好用的结构。
 *
 * 步骤：
 * 1. similaritySearchWithScore(question, k) —— 语义相近的前 k 条
 * 2. 把 [Document, score] 拆成 { score, content, id, book_id, ... }
 * 3. 出错则打日志并返回 []，让上层可以优雅降级（告知「没找到」）
 *
 * @param {string} question 自然语言问题（会经 embeddings 变成向量再搜）
 * @param {number} k        取回几条
 */
async function retrieveRelevantContent(question, k = TOP_K) {
    try {
        // 1. 语义检索：问题 → 向量 → 在 collection 里找近邻
        const docsWithScores = await vectorStore.similaritySearchWithScore(question, k);
        // 2. 展平元数据，缺字段时给「未知」，避免后面拼模板炸掉
        return docsWithScores.map(([doc, score]) => ({
            score,
            content: doc.pageContent,
            id: doc.metadata?.id ?? "unknown",
            book_id: doc.metadata?.book_id ?? "未知",
            chapter_num: doc.metadata?.chapter_num ?? "未知",
            index: doc.metadata?.index ?? "未知",
        }));
    } catch (error) {
        // 3. 降级：检索失败 ≠ 进程崩溃，交给 generate / main 决定怎么说
        console.error("检索内容时出错:", error.message);
        return [];
    }
}

// ============================================================================
// 3. LangGraph 节点：retrieve → generate
// ============================================================================

/**
 * retrieve 节点：只负责「找资料」，不负责回答。
 * 输入 state.question / state.k → 输出更新后的 documents（以及原样带回 question、k）。
 */
const retrieveNode = async (state) => {
    const documents = await retrieveRelevantContent(state.question, state.k);
    return {
        question: state.question,
        k: state.k,
        documents,
    };
};

/**
 * generate 节点：把检索结果拼进提示词，流式调用大模型。
 *
 * 步骤：
 * 1. 把 documents 格式化成带书名/章节的「上下文」字符串
 * 2. 拼 prompt：角色设定 + 上下文 + 用户问题 + 回答要求
 * 3. model.stream 边生成边打印（体验像打字），同时累加到 generation
 * 4. 把完整 generation 写回状态（供 main 做兜底检查）
 *
 * 为什么必须手动拼 context？
 * Naive RAG 的核心就是「检索结果进 prompt」；不拼进去，模型只能靠参数里的常识瞎答。
 */
const generateNode = async (state) => {
    // 1. 把每条片段标号，方便模型引用「片段 1 / 片段 2」
    const context = state.documents
        .map((item, i) => `
            [片段 ${i + 1}]
            书籍: ${item.book_id}
            章节: 第 ${item.chapter_num} 章
            片段索引: ${item.index}
            内容: ${item.content}
        `)
        .join("\n\n━━━━━\n\n");

    // 2. 提示词：约束「只依据片段、没有就如实说」，减少胡编
    const prompt = `
        你是一个专业的《天龙八部》小说助手。基于小说内容回答问题，用准确、详细的语言。
        请根据以下《天龙八部》小说片段内容回答问题：
        ${context}

        用户问题: ${state.question}

        回答要求：
        1. 如果片段中有相关信息，请结合小说内容给出详细、准确的回答
        2. 可以综合多个片段的内容，提供完整的答案
        3. 如果片段中没有相关信息，请如实告知用户
        4. 回答要准确，符合小说的情节和人物设定
        5. 可以引用原文内容来支持你的回答

        AI 助手的回答:
    `;

    // 3. 流式输出：chunk 可能是非字符串 content，空的就跳过
    process.stdout.write("\n【AI 回答（流式）】\n");
    let generation = "";
    const stream = await model.stream(prompt);
    for await (const chunk of stream) {
        const text = typeof chunk.content === "string" ? chunk.content : "";
        if (!text) continue;
        generation += text;
        process.stdout.write(text);
    }
    process.stdout.write("\n");

    // 4. 写回状态（question / k / documents 原样带回，方便 main 打印检索明细）
    return {
        question: state.question,
        k: state.k,
        documents: state.documents,
        generation,
    };
};

// ============================================================================
// 4. 装配图：START → retrieve → generate → END
// ============================================================================

/**
 * 朴素 RAG 的两站流水线：
 * 先 retrieve 填 documents，再 generate 读 documents 写 generation。
 * 没有条件边、没有循环——这就是「naive」的直观形态。
 */
const graph = new StateGraph(GraphState)
    .addNode("retrieve", retrieveNode)
    .addNode("generate", generateNode)
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", END)
    .compile();

// ============================================================================
// 5. 入口 main：连库 → 加载集合 → invoke → 打印检索明细
// ============================================================================

/**
 * 步骤：
 * 1. 准备问题与 k
 * 2.（可选）导出 Mermaid 看图结构
 * 3. 连接已有 Milvus 集合，并配置余弦相似度搜索参数
 * 4. loadCollection（内存里加载，才能搜；已加载则忽略报错）
 * 5. graph.invoke 跑 RAG
 * 6. 打印检索片段；若为空则降级提示（生成阶段可能已流式打过字）
 */
async function main() {
    // 1.
    const question = "阿朱的结局是什么？";
    const kArg = 5;

    // 2. 导出为 Mermaid：可复制到 https://mermaid.live 或 Markdown 的 ```mermaid 代码块
    const drawable = await graph.getGraphAsync();
    const mermaid = drawable.drawMermaid({ withStyles: true });
    console.log(mermaid);

    // 3. 挂接「已有」集合：textField / vectorField 等要和入库时字段名一致
    console.log("连接到 Milvus...");
    vectorStore = await Milvus.fromExistingCollection(embeddings, {
        collectionName: COLLECTION_NAME,
        url: "localhost:19530",
        textField: "content",
        primaryField: "id",
        vectorField: "vector",
        indexCreateOptions: {
            metric_type: "COSINE",
            index_type: "HNSW",
            params: { M: 16, efConstruction: 200 },
            search_params: { ef: 64 },
        },
    });
    // 部分版本还要显式设搜索参数，否则度量/ef 可能对不上入库索引
    vectorStore.indexSearchParams = { metric_type: "COSINE", params: JSON.stringify({ ef: 64 }) };
    console.log("✓ 已连接\n");

    // 4. 集合必须 load 后才能检索；重复 load 会报 already loaded，可忽略
    try {
        await vectorStore.client.loadCollection({ collection_name: COLLECTION_NAME });
        console.log(`✓ 集合 ${COLLECTION_NAME} 已加载\n`);
    } catch (error) {
        if (!error.message.includes("already loaded")) {
            throw error;
        }
        console.log(`✓ 集合 ${COLLECTION_NAME} 已处于加载状态\n`);
    }

    console.log("=".repeat(80));
    console.log(`问题: ${question}`);
    console.log("=".repeat(80));

    // 5. 初始状态：documents / generation 先放空占位，由节点填实
    const result = await graph.invoke({
        question,
        k: Number.isFinite(kArg) ? kArg : TOP_K,
        documents: [],
        generation: "",
    });

    // 6. 检索明细（答案已在 generateNode 里流式打印过）
    console.log("\n【检索相关内容】");
    if (result.documents.length === 0) {
        console.log("未找到相关内容");
        console.log("\n【AI 回答】");
        console.log("抱歉，我没有找到相关的《天龙八部》内容。");
        return;
    } else {
        result.documents.forEach((item, i) => {
            console.log(`\n[片段 ${i + 1}] 相似度: ${item.score.toFixed(4)}`);
            console.log(`书籍: ${item.book_id}`);
            console.log(`章节: 第 ${item.chapter_num} 章`);
            console.log(`片段索引: ${item.index}`);
            console.log(
                `内容: ${item.content.substring(0, 200)}${item.content.length > 200 ? "..." : ""}`,
            );
        });
    }

    // 流式路径异常时 generation 可能仍为空，给个兜底提示
    if (!result.generation) {
        console.log("\n【AI 回答】");
        console.log("模型未返回内容。");
    }
}

main()

/**
 * ----------------------------------------------------------------------------
 * 动手实验（可选）
 * ----------------------------------------------------------------------------
 * 1. 确保本机 Milvus 在 localhost:19530，且 ebook_collection 已入库
 * 2. 运行本文件，应看到：Mermaid 图 → 连接日志 → 流式回答 → 检索片段列表
 * 3. 把 question 改成无关问题（如「今天天气」），观察是否「如实告知没有相关信息」
 * 4. 把 kArg 改成 1 和 10，对比答案完整度与噪声
 * ============================================================================
 */
