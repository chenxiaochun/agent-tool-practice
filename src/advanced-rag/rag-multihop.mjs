/**
 * ============================================================================
 * 新手导读：这份文件在干什么？
 * ============================================================================
 *
 * 目标：演示「多跳 RAG」（Multi-hop）：复杂问题先拆成有序子问题，再一轮轮检索，
 * 攒够证据后再生成答案；简单问题则跳过检索直接回答。
 *
 * 和 naive-rag / rag-query-router 差在哪？
 * - naive-rag        ：一问一检一答（单跳）
 * - rag-query-router ：先路由 simple/complex，complex 仍只检一轮
 * - 本文件           ：complex 会「拆题 → 检 → 规划 → 再检…」循环，像连环追问
 *
 * 生活类比：
 * - 原始问题 = 「四大恶人排行第二是谁？其子揭晓前，生父公开身份是什么？」
 * - 拆解器   = 先列调查提纲：①谁是排行第二？②此人的儿子是谁？③揭晓前生父身份？
 * - 检索     = 按提纲一条条去图书馆抽页（每条子问题一次 similaritySearch）
 * - 规划器   = 看已抽到的页够不够答原题；不够且还有未检子问题 → 继续；够了 → 写答案
 * - documents= 桌上越摊越多的页（按 id 去重，同 id 留更高分）
 *
 * 为什么要多跳？
 * - 一句复杂问法往往塞了多层实体关系；用原句去检，向量库可能只命中「半截」
 * - 先拆成可独立检索的短问，再按推理顺序检，更容易把证据链凑齐
 *
 * 数据前提：Milvus 已有 ebook_collection（与 naive-rag 相同，只读不写）。
 *
 * ----------------------------------------------------------------------------
 * 主流程怎么走？（对应 main + 图节点）
 * ----------------------------------------------------------------------------
 *  1. 定义 GraphState（问题、子问题列表、下标、累计文档、规划结果…）
 *  2. 装配图：
 *       START → route_question
 *         ├─ simple  → direct_answer → END
 *         └─ complex → decompose_question → retrieve ⇄ plan_next_step → generate → END
 *  3. main：连 Milvus → loadCollection → invoke
 *  4. 打印子问题序列、累计片段、最终策略
 *
 * 新手易混点：
 * - 规划器只决定「继续 retrieve / 去 generate」，不会自己改写下一条检索句；
 *   下一条查询永远是 subQuestions[nextSubIdx]
 * - 检索结果不会自动进模型；generate 节点里必须把 documents 拼进 prompt
 * - 当前聊天网关对 withStructuredOutput(json_object) 不友好，本文件用「普通 invoke + 解析 JSON」
 *
 * 阅读建议：先扫导读与图装配（约文件后部 graph = …），再按
 * route → decompose → retrieve → plan → generate 顺序读各节点。
 * ============================================================================
 */

import 'dotenv/config';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { Milvus } from '@langchain/community/vectorstores/milvus';
/** model = 聊天模型；embeddings = 向量模型（见 ../model.mjs） */
import { model, embeddings } from '../model.mjs';

// ============================================================================
// 1. 图状态：多跳流水线共用的「草稿纸」
// ============================================================================

/**
 * 字段怎么在节点间传（复杂路径）：
 * - question / k          ：用户原题与每轮检索条数
 * - strategy / routeReason：路由结果（simple | complex）
 * - subQuestions          ：拆解器写出的有序子问题（只用于检索，不直接当最终答案）
 * - nextSubIdx            ：下一条待检索子问题的下标（retrieve 成功后 +1）
 * - currentQuery          ：本轮实际拿去向量检索的那句（便于日志）
 * - documents             ：多轮合并后的片段池（mergeUnique）
 * - retrievalCount / maxRetrievals：已检轮数 & 硬上限（防死循环）
 * - plannedNext           ：规划器决定的下一步（retrieve | generate）
 * - generation            ：最终回答文本
 */
const GraphState = Annotation.Root({
  question: Annotation,
  k: Annotation,
  strategy: Annotation,
  routeReason: Annotation,
  /** 拆解得到的有序子问题，仅用于检索 */
  subQuestions: Annotation,
  /** 下一轮 retrieve 要用的下标（指向 subQuestions 中尚未检索的那一条） */
  nextSubIdx: Annotation,
  documents: Annotation,
  currentQuery: Annotation,
  retrievalCount: Annotation,
  maxRetrievals: Annotation,
  plannedNext: Annotation,
  generation: Annotation,
});

/**
 * vectorStore：Milvus 客户端，模块级闭包供节点使用。
 * 必须在 main 里赋值并 load 后再 invoke，否则 retrieve 会空引用。
 */
let vectorStore;

// ============================================================================
// 2. 工具函数：检索 & 去重合并
// ============================================================================

/**
 * 对当前查询做向量相似度检索。
 * 步骤：1. similaritySearchWithScore  2. 压成统一结构（含 score / 元数据）
 */
async function retrieveRelevantContent(question, k) {
  try {
    // 1. 问句 → 向量 → 在书架上找最近的 k 页（带分数）
    const docsWithScores = await vectorStore.similaritySearchWithScore(
      question,
      k
    );
    // 2. 统一字段，方便后面 merge / 拼 prompt / 打印
    return docsWithScores.map(([doc, score]) => ({
      score,
      content: doc.pageContent,
      id: doc.metadata?.id ?? 'unknown',
      book_id: doc.metadata?.book_id ?? '未知',
      chapter_num: doc.metadata?.chapter_num ?? '未知',
      index: doc.metadata?.index ?? '未知',
    }));
  } catch (error) {
    console.error('检索内容时出错:', error.message);
    return [];
  }
}

/**
 * 按 id 合并多轮检索结果；同 id 保留更高 score。
 * 为什么需要：多跳会重复命中同一片段，直接 concat 会膨胀且浪费 token。
 */
function mergeUnique(existingDocs, newDocs) {
  const map = new Map();
  for (const d of [...existingDocs, ...newDocs]) {
    const key = String(d.id);
    const prev = map.get(key);
    if (!prev || Number(d.score) > Number(prev.score)) {
      map.set(key, d);
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => Number(b.score) - Number(a.score)
  );
}

/**
 * 从模型回复里抽出 JSON 对象。
 * 为什么不用 withStructuredOutput：部分网关要求 messages 含 "json" 字样，且结构化输出易 400。
 */
function parseJsonObject(raw) {
  const text = String(raw ?? '').trim();
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    return JSON.parse(match[0]);
  } catch {
    return null;
  }
}

// ============================================================================
// 3. 节点：路由 —— 要不要走多跳检索？
// ============================================================================

/**
 * route_question：判断 simple（直答）还是 complex（拆题多跳）。
 * 步骤：1. 调模型拿 JSON  2. 解析 strategy  3. 重置多跳相关状态
 */
const routeQuestionNode = async (state) => {
  console.log('---ROUTE_QUESTION---');
  // 1. 普通 invoke；提示里显式要求 JSON，兼容当前网关
  const response = await model.invoke(`
你是问答路由器。请判断用户问题是否需要外部检索，并用 JSON 返回结果。

规则：
- simple: 常识问答、简短定义、无需特定小说细节即可回答。
- complex: 需要《天龙八部》具体情节、人物关系、章节事实、原文细节或证据支持。

只输出一行 JSON，格式严格为：{"strategy":"simple|complex","reason":"一句话理由"}

用户问题：${state.question}
`);
  // 2. 解析；失败则偏 complex，避免该检索时误走直答
  const parsed = parseJsonObject(response.content);
  const strategy = parsed?.strategy === 'simple' ? 'simple' : 'complex';
  const reason = String(
    parsed?.reason ?? response.content ?? '无法解析路由，默认 complex'
  );

  console.log(`路由策略: ${strategy} (${reason})`);
  // 3. 清空上一轮残留，给多跳路径一个干净起点
  return {
    strategy,
    routeReason: reason,
    retrievalCount: 0,
    maxRetrievals: state.maxRetrievals ?? 8,
    documents: [],
    subQuestions: [],
    nextSubIdx: 0,
    currentQuery: '',
  };
};

// ============================================================================
// 4. 节点：拆解 —— 把原题拆成「可独立检索」的有序子问题
// ============================================================================

/**
 * decompose_question：只在 complex 路径执行。
 * 关键：每条必须写全人名/事件名，禁止「他/她」——否则向量检索对不上实体。
 * 步骤：1. 模型拆题  2. 清洗列表  3. 把第 0 条设为即将检索的 currentQuery
 */
const decomposeQuestionNode = async (state) => {
  console.log('---DECOMPOSE_QUESTION---');
  const response =
    await model.invoke(`你是《天龙八部》多跳问答的「子问题拆解器」。请用 JSON 返回结果。

用户原始问题：
${state.question}

任务：将问题拆成**有序**子问题列表 sub_questions，用于**依次向量检索**。要求：
1. 链式推理、多层关系、因果先后的问题，必须拆成多条；单跳即可答的也可只输出 1 条。
2. 每条子问题必须是**可独立检索**的完整中文问句，**禁止**使用「他/她/此人/上文」等指代；可写全人物名与事件名。
3. 顺序必须符合推理链：先搞清前置实体/事实，再查后续结论。
4. **不要**把整句原题原样复制成唯一一条（除非确实无法拆分）；不要拆成过碎的关键词列表。
5. 输出 1～8 条即可。

只输出一行 JSON，格式严格为：{"sub_questions":["问句1","问句2"],"reason":"一句话理由"}`);

  const parsed = parseJsonObject(response.content);
  const subQuestions = (parsed?.sub_questions ?? [])
    .map((s) => String(s).trim())
    .filter(Boolean);
  if (subQuestions.length === 0) {
    throw new Error('decompose_question: sub_questions 为空');
  }

  console.log(`拆解 ${subQuestions.length} 条子问题 (${parsed?.reason ?? ''})`);
  subQuestions.forEach((q, i) => {
    console.log(`  [${i + 1}] ${q}`);
  });

  return {
    subQuestions,
    nextSubIdx: 0,
    // 第一条立刻成为 retrieve 的查询句
    currentQuery: subQuestions[0],
  };
};

// ============================================================================
// 5. 节点：检索 —— 用「当前子问题」去向量库抽一页，并入累计池
// ============================================================================

/**
 * retrieve：按 nextSubIdx 取子问题 → 检索 → 合并 → 下标 +1。
 * 步骤：1. 取出第 idx 条子问题  2. 向量检索  3. mergeUnique  4. 推进下标与轮数
 */
const retrieveNode = async (state) => {
  const subs = state.subQuestions ?? [];
  const idx = state.nextSubIdx ?? 0;
  // 1. 本轮查询句必须来自拆解列表，不由本节点自由发挥
  const q = subs[idx]?.trim();
  if (!q) {
    throw new Error(
      `retrieve: 子问题下标 ${idx} 无有效文本（共 ${subs.length} 条）`
    );
  }

  const round = state.retrievalCount + 1;
  console.log(
    `---RETRIEVE (第 ${round} 轮，子问题 ${idx + 1}/${subs.length})---`
  );
  console.log(`查询: ${q}`);

  // 2～3. 检索并入累计文档池
  const newDocs = await retrieveRelevantContent(q, state.k);
  const merged = mergeUnique(state.documents ?? [], newDocs);

  if (newDocs.length === 0) {
    console.log('本轮未命中文档');
  } else {
    console.log(
      `本轮命中 ${newDocs.length} 条，累计去重后 ${merged.length} 条`
    );
    newDocs.forEach((item, i) => {
      const preview =
        item.content.length > 120
          ? `${item.content.substring(0, 120)}...`
          : item.content;
      console.log(
        `[R${i + 1}] score=${Number(item.score).toFixed(4)} chapter=${item.chapter_num} index=${item.index}`
      );
      console.log(`      ${preview}`);
    });
  }

  // 4. 下标前进：下一轮若再 retrieve，就会用下一条子问题
  return {
    documents: merged,
    retrievalCount: round,
    nextSubIdx: idx + 1,
    currentQuery: q,
  };
};

// ============================================================================
// 6. 节点：规划 —— 证据够了就生成，否则继续下一条子问题
// ============================================================================

/**
 * plan_next_step：看「剩余子问题 + 已有文档 + 轮数上限」决定下一步。
 * 易混点：模型只选 retrieve/generate，不能发明新检索句；硬性规则会覆盖模型建议。
 * 步骤：1. 拼现状给模型  2. 解析 nextAction  3. 用硬规则钳制  4. 写入 plannedNext
 */
const planNextStepNode = async (state) => {
  console.log('---PLAN_NEXT_STEP---');
  const subs = state.subQuestions ?? [];
  const nextIdx = state.nextSubIdx ?? 0;
  const remaining = subs.length - nextIdx;

  // 1. 把子问题进度 & 已召回摘要摊给规划器（摘要截断，省 token）
  const subList = subs
    .map(
      (s, i) =>
        `${i + 1}. ${s}${i < nextIdx ? ' （已检索）' : i === nextIdx ? ' （下一轮将检索，若选择继续）' : ' （未检索）'}`
    )
    .join('\n');

  // 已召回文档摘要，截断，省 token
  const docStr =
    state.documents.length === 0
      ? '（尚无检索结果）'
      : state.documents
          .slice(0, 6)
          .map(
            (d, i) =>
              `[${i + 1}] score=${Number(d.score).toFixed(4)} 第${d.chapter_num}章: ${d.content.slice(0, 200)}${d.content.length > 200 ? '...' : ''}`
          )
          .join('\n\n');

  // 拼 prompt, 提示模型下一步要做什么, 不要自拟新的检索句。
  const prompt = `你是多跳 RAG 规划器。检索查询已由前置步骤拆解为**有序子问题**；若需继续检索，下一轮将自动使用「下一条子问题」做向量检索，你**不要**自拟新的检索句。请用 JSON 返回结果。

用户原始问题：${state.question}

子问题序列：
${subList || '（无）'}

已检索轮数：${state.retrievalCount}；剩余未检索子问题条数：${remaining}
最大检索轮数上限：${state.maxRetrievals}

已召回文档摘要：
${docStr}

请判断下一步：
1) 已有足够依据回答用户原始问题 → nextAction=generate
2) 仍缺关键事实、且仍存在未检索的子问题、且未超过轮数上限 → nextAction=retrieve

硬性规则：
- 若剩余未检索子问题条数为 0，必须 nextAction=generate。
- 若已检索轮数已达到或超过最大检索轮数，必须 nextAction=generate。

只输出一行 JSON，格式严格为：{"nextAction":"retrieve|generate","reason":"一句话理由"}`;

  // 2. 注意：不要写 const model = model....（会遮蔽导入并触发暂时性死区）
  const response = await model.invoke(prompt);
  const parsed = parseJsonObject(response.content);
  const nextAction =
    parsed?.nextAction === 'retrieve' ? 'retrieve' : 'generate';
  const reason = String(parsed?.reason ?? response.content ?? '');

  // 3. 硬规则兜底：没子问题可检 / 触达轮数上限 → 强制生成
  let finalNext = nextAction;
  if (state.retrievalCount >= state.maxRetrievals) finalNext = 'generate';
  if (remaining <= 0) finalNext = 'generate';

  console.log(
    `[决策] plannedNext=${finalNext} (模型建议=${nextAction}) (${reason})`
  );

  // 4. 条件边 afterPlan 只读 plannedNext
  return {
    plannedNext: finalNext,
  };
};

/** 路由后分支：简单直答 vs 复杂拆题 */
function afterRoute(state) {
  return state.strategy === 'simple' ? 'direct_answer' : 'decompose_question';
}

/** 规划后分支：再检索一跳 vs 去生成 */
function afterPlan(state) {
  return state.plannedNext === 'retrieve' ? 'retrieve' : 'generate';
}

// ============================================================================
// 7. 节点：回答 —— 直答 / 基于累计片段生成
// ============================================================================

/** simple 路径：不检索，模型直接答 */
const directAnswerNode = async (state) => {
  console.log('---DIRECT_ANSWER---');
  process.stdout.write('\n【AI 回答（流式）】\n');
  let generation = '';
  const stream = await model.stream(`你是一个中文问答助手，请直接简洁回答问题。

问题：${state.question}
`);
  for await (const chunk of stream) {
    const text = typeof chunk.content === 'string' ? chunk.content : '';
    if (!text) continue;
    generation += text;
    process.stdout.write(text);
  }
  process.stdout.write('\n');
  return { generation };
};

/**
 * generate：把累计 documents 拼进 prompt 再答原题。
 * 易混点：这里答的是用户原始 question，不是某一条子问题。
 */
const generateNode = async (state) => {
  console.log('---GENERATE---');
  const context = state.documents
    .map(
      (item, i) =>
        `[片段 ${i + 1}]
章节: 第 ${item.chapter_num} 章
内容: ${item.content}`
    )
    .join('\n\n━━━━━\n\n');
  process.stdout.write('\n【AI 回答（流式）】\n');
  let generation = '';
  const stream =
    await model.stream(`你是一个专业的《天龙八部》小说助手。基于小说内容回答问题，用准确、详细的语言。

请根据以下《天龙八部》小说片段内容回答问题：
${context || '（未检索到相关内容）'}

用户问题: ${state.question}

回答要求：
1. 如果片段中有相关信息，请结合小说内容给出详细、准确的回答
2. 可以综合多个片段的内容，提供完整的答案
3. 如果片段中没有相关信息，请如实告知用户
4. 回答要准确，符合小说的情节和人物设定
5. 可以引用原文内容来支持你的回答

AI 助手的回答:`);
  for await (const chunk of stream) {
    const text = typeof chunk.content === 'string' ? chunk.content : '';
    if (!text) continue;
    generation += text;
    process.stdout.write(text);
  }
  process.stdout.write('\n');
  return { generation };
};

// ============================================================================
// 8. 装配图：把节点用边焊成可运行机器
// ============================================================================

/**
 * 拓扑（complex 主路径）：
 *   route → decompose → retrieve → plan ─┬─ retrieve（环）
 *                                        └─ generate → END
 * simple：route → direct_answer → END
 */
const graph = new StateGraph(GraphState)
  .addNode('route_question', routeQuestionNode)
  .addNode('direct_answer', directAnswerNode)
  .addNode('decompose_question', decomposeQuestionNode)
  .addNode('retrieve', retrieveNode)
  .addNode('plan_next_step', planNextStepNode)
  .addNode('generate', generateNode)
  .addEdge(START, 'route_question')
  .addConditionalEdges('route_question', afterRoute, {
    direct_answer: 'direct_answer',
    decompose_question: 'decompose_question',
  })
  .addEdge('decompose_question', 'retrieve')
  .addEdge('retrieve', 'plan_next_step')
  .addConditionalEdges('plan_next_step', afterPlan, {
    retrieve: 'retrieve',
    generate: 'generate',
  })
  .addEdge('direct_answer', END)
  .addEdge('generate', END)
  .compile();

// ============================================================================
// 9. 入口：连库 → 跑图 → 打印证据与策略
// ============================================================================

async function main() {
  // 故意选「多层关系」题，方便观察拆题 + 多轮检索
  const question =
    '《天龙八部》中「四大恶人」排行第二的是谁？此人之子在身世揭晓前，其生父在武林中的公开身份是什么？';
  const k = 5;

  // 9.1 导出 Mermaid，对照节点与环
  const drawable = await graph.getGraphAsync();
  console.log(drawable.drawMermaid({ withStyles: true }));

  // 9.2 连接已有集合（只读）
  console.log('连接到 Milvus...');
  vectorStore = await Milvus.fromExistingCollection(embeddings, {
    collectionName: 'ebook_collection',
    url: 'localhost:19530',
    textField: 'content',
    primaryField: 'id',
    vectorField: 'vector',
    indexCreateOptions: {
      metric_type: 'COSINE',
      index_type: 'HNSW',
      params: { M: 16, efConstruction: 200 },
      search_params: { ef: 64 },
    },
  });
  vectorStore.indexSearchParams = {
    metric_type: 'COSINE',
    params: JSON.stringify({ ef: 64 }),
  };
  console.log('✓ 已连接\n');

  // 9.3 检索前集合必须 load
  try {
    await vectorStore.client.loadCollection({
      collection_name: 'ebook_collection',
    });
    console.log('✓ 集合 ebook_collection 已加载\n');
  } catch (error) {
    if (!error.message.includes('already loaded')) {
      throw error;
    }
    console.log('✓ 集合 ebook_collection 已处于加载状态\n');
  }

  console.log('='.repeat(80));
  console.log(`问题: ${question}`);
  console.log('='.repeat(80));

  // 9.4 塞初始状态，跑完整条图（含可能的 retrieve 环）
  const result = await graph.invoke({
    question,
    k: Number.isFinite(k) ? k : 5,
    strategy: '',
    routeReason: '',
    subQuestions: [],
    nextSubIdx: 0,
    documents: [],
    currentQuery: '',
    retrievalCount: 0,
    maxRetrievals: 8,
    plannedNext: '',
    generation: '',
  });

  // 9.5 复杂路径：回放子问题与累计证据，便于对照模型回答
  if (result.strategy === 'complex') {
    if (result.subQuestions?.length) {
      console.log('\n【子问题序列】');
      result.subQuestions.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
    }
    console.log('\n【检索相关内容（累计）】');
    if (result.documents.length === 0) {
      console.log('未找到相关内容');
    } else {
      result.documents.forEach((item, i) => {
        console.log(
          `\n[片段 ${i + 1}] 相似度: ${Number(item.score).toFixed(4)}`
        );
        console.log(`书籍: ${item.book_id}`);
        console.log(`章节: 第 ${item.chapter_num} 章`);
        console.log(`片段索引: ${item.index}`);
        console.log(
          `内容: ${item.content.substring(0, 200)}${item.content.length > 200 ? '...' : ''}`
        );
      });
    }
    console.log(
      `\n检索轮数: ${result.retrievalCount} / ${result.maxRetrievals}`
    );
  }

  console.log(`\n最终策略: ${result.strategy}`);
  if (!result.generation?.trim()) {
    console.log('模型未返回内容。');
  }
}

/**
 * ----------------------------------------------------------------------------
 * 动手实验（可选）
 * ----------------------------------------------------------------------------
 * 1. 运行：node src/advanced-rag/rag-multihop.mjs
 * 2. 看控制台是否出现 DECOMPOSE → 多轮 RETRIEVE → PLAN → GENERATE
 * 3. 把 question 改成「什么是 RAG？」之类常识题，预期走 simple → DIRECT_ANSWER
 * 4. 把 maxRetrievals 改成 1，观察是否提早结束检索去生成
 * ----------------------------------------------------------------------------
 */

main().catch((err) => {
  console.error('运行失败:', err);
  process.exit(1);
});
