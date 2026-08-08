/**
 * ============================================================================
 * 新手导读：这份文件在干什么？
 * ============================================================================
 *
 * 目标：演示「混合检索 Hybrid Retrieval」流水线——同一句用户问题，
 * 一边走 Elasticsearch（关键词/全文），一边走 Milvus（语义向量），
 * 合并去重后用 DashScope 重排，再让大模型只依据保留下来的笔记片段作答。
 *
 * 生活类比：
 * - ES        = 按「字面关键词」在目录里翻（断流、无线、滤芯…）
 * - Milvus    = 按「意思像不像」在书架上抽（口语「断断续续」≈ 路由器断流）
 * - 查询扩展  = 让 LLM 再写 3 句同义检索问句，扩大召回面
 * - merge     = 两堆纸条按笔记 id 去重，摊成一桌
 * - rerank    = 专业导购按「和原问题有多贴」重排，只留 topN（默认 3）
 * - generate  = 专家对着桌上留下的页回答，禁止瞎编库外事实
 *
 * 为什么要混合？
 * - 纯关键词：口语/同义说法容易漏（「断断续续」未必命中「路由器」）
 * - 纯向量：专有名词、订单号有时不如字面匹配稳
 * - 两边召回 + 重排，通常比单路更稳
 *
 * 对照一次真实终端输出（query = 「家里无线老是断断续续的咋整啊」）：
 *  1. Mermaid 图：START → query_augment →(并行) es_recall & milvus_recall
 *                 → merge → rerank → generate_answer → END
 *  2. 查询扩展：理想情况是 3 条不同改写；若模型失败会用原句填满（见下方「易混点」）
 *  3. ES 可能召回「租房合同 / 净水器」等字面相关但不贴题的笔记
 *  4. Milvus 更容易召回「路由器偶尔断流排查笔记」等语义相关笔记
 *  5. 重排后保留 3 条：断流排查 > 酒店网速 > （可能夹杂弱相关）
 *  6. 最终回答基本复述重排第 1 条里的排查步骤
 *
 * 数据前提：
 * - ES 索引 / Milvus 集合名均为 life_notes（先跑 seed-data.mjs 灌库）
 * - docker-compose 起好 ES、Milvus；.env 配好模型与重排 API
 *
 * ----------------------------------------------------------------------------
 * 主流程怎么走？（对应 compileHybridRetrievalGraph + 文末 invoke）
 * ----------------------------------------------------------------------------
 *  1. 定义 HybridRetrievalState（草稿纸字段）
 *  2. 装配图节点：扩展 → 双路召回 → 合并 → 重排 → 生成
 *  3. 连接 ES / Milvus 客户端
 *  4. 对 SAMPLE_QUERIES 逐条 invoke，并用 chalk 打印各阶段结果
 *
 * 新手易混点：
 * - query_augment 失败时 normalizeThreeQueries 会用「原句」填满 3 格，
 *   终端会看到 3～4 条完全一样的检索串——不是打印 bug，是扩展失败/退化
 * - 检索结果不会自动进模型；generate_answer 必须把 topDocuments 拼进 prompt
 * - merge 按 metadata.id 去重，无 id 的文档会被丢掉
 *
 * 阅读建议：先看文末 Mermaid / for 循环打印，再按节点顺序读 compileHybridRetrievalGraph。
 * ============================================================================
 */

import 'dotenv/config';
import chalk from 'chalk';
import { Client } from '@elastic/elasticsearch';
import { Document } from '@langchain/core/documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Milvus } from '@langchain/community/vectorstores/milvus';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { augmentQuery, retrievalQueryStrings } from './query-argument.mjs';
import { embeddings } from '../model.mjs';
import { model } from '../model.mjs';
import { DashscopeRerank } from './dashscope-rerank.mjs';

// ============================================================================
// 配置：索引名、控制台配色、重排器
// ============================================================================

/** ES 索引名 & Milvus 集合名（seed-data 写入的生活笔记库） */
const INDEX = 'life_notes';

/** 各阶段控制台颜色：一眼区分扩展 / ES / Milvus / 重排 / 生成 */
const log = {
  setup: chalk.gray,
  query: chalk.cyan,
  augment: chalk.magenta,
  es: chalk.yellow,
  milvus: chalk.blue,
  rerank: chalk.whiteBright,
  answer: chalk.green,
  meta: chalk.dim,
  err: chalk.red,
};

/** DashScope 文本重排；默认 topN=3，对应终端「重排后保留 (3 条)」 */
const reranker = new DashscopeRerank();

// ============================================================================
// 1. 图状态：整条混合检索流水线共用的「草稿纸」
// ============================================================================

/**
 * 字段谁写谁读：
 * - query              ：用户原句（入口写入；扩展/重排/生成都读）
 * - queryAugmentation  ：query_augment 写出 { queries: [q1,q2,q3] }
 * - esHits / milvusHits：两路召回各自写出 Document[]
 * - merged             ：merge 去重后的合集
 * - topDocuments       ：rerank 留下的 topN
 * - answer             ：generate_answer 最终回答文本
 */
const HybridRetrievalState = Annotation.Root({
  query: Annotation(),
  queryAugmentation: Annotation(),
  esHits: Annotation(),
  milvusHits: Annotation(),
  merged: Annotation(),
  topDocuments: Annotation(),
  answer: Annotation(),
});

// ============================================================================
// 2. 工具：ES hit → Document、合并去重、彩色打印
// ============================================================================

/**
 * 把 ES 的一条 hit 收成 LangChain Document。
 *
 * 为什么要「置换」成这种形状？
 * - ES 原始结构是 _id / _source.note_title / note_body……
 * - Milvus / 重排 / 生成 认的是 { pageContent, metadata }
 * - 两边字段名不同，不统一就无法 merge、rerank、拼进 prompt
 *
 * 各字段怎么填：
 * - pageContent = 标题 + 正文：重排和生成要读「给人看的整段」；标题常含关键信息，不能丢
 * - metadata.id：与 Milvus 命中按同一笔记去重
 * - metadata.source:'es'：事后能区分这条是关键词召回还是向量召回
 * - ...s：保留原始 _source（tags / mood 等），便于打印排查
 */
function docFromEsHit(hit) {
  const s = hit._source ?? {};
  const text = [s.note_title ?? s.title, s.note_body ?? s.content]
    .filter(Boolean)
    .join('\n');
  return new Document({
    pageContent: text,
    metadata: { id: hit._id, source: 'es', ...s },
  });
}

/**
 * ES 结果在前、Milvus 在后拼接，再按 id 去重。
 * 同 id 保留「先出现的那条」——因此 ES 命中通常优先于向量命中。
 */
function merge(esDocs, milvusDocs) {
  const combined = [...(esDocs ?? []), ...(milvusDocs ?? [])].filter(
    (d) => d?.pageContent
  );
  return dedupeDocsById(combined);
}

/**
 * 去重键仅为 metadata.id（trim 后非空）。
 * 无 id → 丢弃；已见过的 id → 跳过；顺序 = 首次出现顺序。
 */
function dedupeDocsById(docs) {
  const seen = new Set();
  const out = [];
  for (const d of docs ?? []) {
    if (!d?.pageContent) continue;
    const id = d.metadata?.id != null ? String(d.metadata.id).trim() : '';
    if (!id) continue;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(d);
  }
  return out;
}

/**
 * 打印一批 Document：标题用阶段色，正文预览同色，metadata 用 dim。
 * @param {'es'|'milvus'|'rerank'} source 决定 chalk 配色
 */
function printDocs(label, docs, source = 'rerank') {
  const paint =
    source === 'es' ? log.es : source === 'milvus' ? log.milvus : log.rerank;
  console.log(paint.bold(`\n=== ${label} (${docs?.length ?? 0} 条) ===`));
  for (let i = 0; i < (docs ?? []).length; i++) {
    const d = docs[i];
    const preview = (d.pageContent ?? '').slice(0, 200).replace(/\n/g, ' ');
    console.log(
      paint(`[${i}] ${preview}${d.pageContent?.length > 200 ? '…' : ''}`)
    );
    console.log(log.meta(`    metadata:`), d.metadata ?? {});
  }
}

/**
 * 打印查询扩展结果。
 * - qs：LLM 声称生成的 3 条
 * - forRetrieval：真正拿去检索的列表 = [原句, ...qs]（见 retrievalQueryStrings）
 * 若终端里 4 条完全相同，说明扩展退化成了「原句填空」。
 */
/**
 * 只负责「打印」查询扩展结果，方便对照终端；不改 state，也不发检索请求。
 *
 * 你会看到两块列表，别混：
 * 1) qs（augmentation.queries）
 *    —— LLM 另外写出的那 3 条改写问句（理想情况应互不相同）
 * 2) forRetrieval（retrievalQueryStrings）
 *    —— 真正拿去打 ES / Milvus 的完整列表 = [原始问题, ...qs]
 *       所以条数通常是 1+3=4；每条都会各搜一次两边
 *
 * 为什么要单独打出来？
 * - 扩展失败时 qs 会退化成「原句×3」，forRetrieval 就变成 4 条一模一样
 *   —— 一眼能看出「扩展没生效」，而不是误以为检索逻辑写错了
 */
function printQueryRewrite(original, augmentation) {
  // LLM 声称生成的 3 条改写
  const qs = augmentation?.queries ?? [];
  // 实际检索串：原句在前 + 上面 3 条（见 query-argument.mjs）
  const forRetrieval = retrievalQueryStrings(original, augmentation);

  console.log(
    log.augment.bold(`\n--- 查询扩展（LLM 生成 ${qs.length} 条检索问句）---`)
  );
  console.log(log.augment('原始 query:'), log.query(original ?? ''));
  for (let i = 0; i < qs.length; i++) {
    console.log(log.augment(`  [${i + 1}] ${qs[i] ?? ''}`));
  }
  console.log(
    log.augment(
      `\n逐条 ES + Milvus（共 ${forRetrieval.length} 条检索串，含原始问题）:`
    )
  );
  for (let i = 0; i < forRetrieval.length; i++) {
    console.log(log.augment(`  [${i + 1}] ${forRetrieval[i] ?? ''}`));
  }
}

/** 把模型 message.content（字符串或分段数组）收成纯文本 */
function stringifyMessageContent(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return String(content ?? '');
  return content
    .map((c) =>
      typeof c === 'string' ? c : typeof c?.text === 'string' ? c.text : ''
    )
    .join('');
}

/**
 * 把 topDocuments 拼成生成节点的「检索片段」上下文。
 * 没有这一步，重排结果进不了模型——这是 RAG 最常见的新手坑。
 */
function formatDocsAsContext(docs) {
  return (docs ?? [])
    .map((d, i) => {
      const meta = d.metadata ?? {};
      const src = meta.source ?? '';
      const id = meta.id != null ? String(meta.id) : '';
      const head = id
        ? `[${i + 1}] id=${id}${src ? ` source=${src}` : ''}`
        : `[${i + 1}]`;
      return `${head}\n${d.pageContent ?? ''}`;
    })
    .join('\n\n---\n\n');
}

// ============================================================================
// 3. 生成用 Prompt：有上下文 / 无上下文两套
// ============================================================================

/** 有 topDocuments 时：只根据片段答，不编造 */
const ANSWER_PROMPT = ChatPromptTemplate.fromMessages([
  [
    'system',
    `你是阅读用户「生活笔记」知识库并作答的助手。
  规则：
  - 只根据下方「检索片段」推断答案；片段里没有的信息不要编造。
  - 若片段不足以回答，明确说明「笔记里未提到」，并可给出一句保守建议。
  - 回答简洁有条理，可使用简短列表；口吻自然中文。`,
  ],
  [
    'human',
    `用户问题：{query}
  
  检索片段：
  {context}`,
  ],
]);

/** 重排后一条都没有时：礼貌说明无法从笔记回答 */
const NO_CONTEXT_PROMPT = ChatPromptTemplate.fromMessages([
  [
    'system',
    `你是阅读用户「生活笔记」知识库并作答的助手。当前没有检索到任何片段。
  请用一两句话说明无法从笔记中回答，并礼貌询问用户是否换个说法或补充关键词。`,
  ],
  ['human', '用户问题：{query}'],
]);

// ============================================================================
// 4. 装配 LangGraph：扩展 → 双路并行召回 → 合并 → 重排 → 生成
// ============================================================================

/**
 * 编译可 invoke 的混合检索图。
 *
 * 拓扑（与终端 Mermaid 一致）：
 *   START → query_augment ┬→ es_recall ──┐
 *                         └→ milvus_recall┤→ merge → rerank → generate_answer → END
 *
 * ES_K / MILVUS_K：两路各自「总召回预算」；会按检索串条数均分到每条 query。
 */
export function compileHybridRetrievalGraph(esClient, milvus, reranker, model) {
  const ES_K = 15;
  const MILVUS_K = 15;

  return (
    new StateGraph(HybridRetrievalState)
      // ---------- 4.1 查询扩展：原句 → 再写 3 条检索问句 ----------
      .addNode('query_augment', async (state) => ({
        queryAugmentation: await augmentQuery(model, state.query ?? ''),
      }))
      // ---------- 4.2 ES 关键词召回（可与 milvus 并行） ----------
      .addNode('es_recall', async (state) => {
        // 1. 取出「原句 + 扩展句」列表
        const qs = retrievalQueryStrings(state.query, state.queryAugmentation);
        const n = Math.max(1, qs.length);
        // 2. 总预算 ES_K 均分到每条检索串（至少 2）
        const kEach = Math.max(2, Math.ceil(ES_K / n));
        // 3. 每条问句各搜一次 multi_match（标题权重 ^2，IK 分词）
        const batches = await Promise.all(
          qs.map((q) =>
            esClient.search({
              index: INDEX,
              size: kEach,
              query: {
                multi_match: {
                  query: q,
                  fields: ['note_title^2', 'note_body', 'title', 'content'],
                  type: 'best_fields',
                  analyzer: 'ik_smart',
                },
              },
            })
          )
        );
        // 4. 摊平 → Document → 按 id 去重
        const flat = batches.flatMap((res) =>
          (res.hits?.hits ?? []).map(docFromEsHit)
        );
        return { esHits: dedupeDocsById(flat) };
      })
      // ---------- 4.3 Milvus 语义召回（与 ES 并行） ----------
      .addNode('milvus_recall', async (state) => {
        const qs = retrievalQueryStrings(state.query, state.queryAugmentation);
        const n = Math.max(1, qs.length);
        const kEach = Math.max(2, Math.ceil(MILVUS_K / n));
        // 每条问句 embedding 后做相似度搜索
        const batches = await Promise.all(
          qs.map((q) => milvus.similaritySearch(q, kEach))
        );
        const flat = batches.flat();
        return { milvusHits: dedupeDocsById(flat) };
      })
      // ---------- 4.4 合并去重（等两边都跑完才进 merge） ----------
      .addNode('merge', async (state) => ({
        merged: merge(state.esHits, state.milvusHits),
      }))
      // ---------- 4.5 重排：按与「用户原句」相关性筛 topN ----------
      .addNode('rerank', async (state) => {
        const merged = state.merged ?? [];
        if (!merged.length) return { topDocuments: [] };
        // compressDocuments：传入候选 docs + 原 query，返回重排后的子集
        const topDocuments = await reranker.compressDocuments(
          merged,
          state.query
        );
        return { topDocuments };
      })
      // ---------- 4.6 生成：把 topDocuments 塞进 prompt 再答 ----------
      .addNode('generate_answer', async (state) => {
        const query = state.query ?? '';
        const docs = state.topDocuments ?? [];
        if (!docs.length) {
          const chain = NO_CONTEXT_PROMPT.pipe(model);
          const msg = await chain.invoke({ query });
          return { answer: stringifyMessageContent(msg.content).trim() };
        }
        const chain = ANSWER_PROMPT.pipe(model);
        const msg = await chain.invoke({
          query,
          context: formatDocsAsContext(docs),
        });
        return { answer: stringifyMessageContent(msg.content).trim() };
      })
      .addEdge(START, 'query_augment')
      // 扩展后同时进两路召回（LangGraph 扇出）
      .addEdge('query_augment', 'es_recall')
      .addEdge('query_augment', 'milvus_recall')
      // 两边都完成才进 merge（扇入）
      .addEdge(['es_recall', 'milvus_recall'], 'merge')
      .addEdge('merge', 'rerank')
      .addEdge('rerank', 'generate_answer')
      .addEdge('generate_answer', END)
      .compile()
  );
}

// ============================================================================
// 5. 入口：连库 → 画图 → 对示例问题跑完整条流水线并打印
// ============================================================================

const esClient = new Client({ node: 'http://localhost:9200' });

const milvus = await Milvus.fromExistingCollection(embeddings, {
  url: 'http://localhost:19530',
  collectionName: INDEX,
  textField: 'doc_text',
  vectorField: 'embedding',
});

/**
 * 示例用户 query。
 * 「家里无线老是断断续续」适合观察：ES 可能偏字面噪声，Milvus + 重排拉回路由器笔记。
 */
const SAMPLE_QUERIES = [
  // "PO-20250409-K9 滤芯订单",
  // '家里无线老是断断续续的咋整啊',
  // "那个黑凉粉粉怎么冲不结块",
  '明火炖太久汤汁又黏又涩，起锅前要怎么处理才不腻',
];

const graph = compileHybridRetrievalGraph(esClient, milvus, reranker, model);

// 5.1 导出 Mermaid：对照节点名与并行边（可粘到 https://mermaid.live）
const drawable = await graph.getGraphAsync();
console.log(log.setup(drawable.drawMermaid()));
console.log();

// 5.2 逐条 invoke，并按阶段彩色打印（对应终端各 === 区块）
for (const query of SAMPLE_QUERIES) {
  console.log(log.query.bold(`query: ${query}`));

  // 跑完一整图；state 里已有扩展 / 两路命中 / 重排 / 回答
  const state = await graph.invoke({ query });

  // 扩展结果（含退化成原句重复的情况）
  printQueryRewrite(state.query, state.queryAugmentation);
  console.log(
    log.meta('\n（原始 JSON）'),
    log.meta(JSON.stringify(state.queryAugmentation))
  );

  // 两路原始召回 vs 重排后留下的上下文
  printDocs('Elasticsearch 检索', state.esHits, 'es');
  printDocs('Milvus 检索', state.milvusHits, 'milvus');
  printDocs('重排后保留', state.topDocuments ?? [], 'rerank');

  console.log(log.answer.bold('\n=== 大模型生成回答 ===\n'));
  console.log(log.answer(state.answer ?? log.err('（空）')));
}

/**
 * ----------------------------------------------------------------------------
 * 动手实验（可选）
 * ----------------------------------------------------------------------------
 * 1. 运行：node src/es-test/hybrid-retrieval.mjs
 * 2. 看 Mermaid 是否画出 query_augment 扇出到 es_recall / milvus_recall
 * 3. 对比「ES 检索」与「Milvus 检索」命中是否不同；看重排是否把路由器笔记顶到第 1
 * 4. 若查询扩展三条全是原句：检查 withStructuredOutput / 网关是否报错被 catch 吞掉
 * 5. 换 SAMPLE_QUERIES 里带订单号的句子，观察 ES 字面匹配是否更强
 * ----------------------------------------------------------------------------
 */
