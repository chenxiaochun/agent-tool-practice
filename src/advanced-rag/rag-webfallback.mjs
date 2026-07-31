/**
 * ============================================================================
 * 新手导读：这份文件在干什么？
 * ============================================================================
 *
 * 目标：演示「RAG + 联网兜底」（Web Fallback）：先查本地知识库；不够再去网上搜一轮，
 * 把本地证据 + 联网证据一起交给模型作答；简单常识题则跳过检索直接回答。
 *
 * 和其它 advanced-rag 脚本差在哪？
 * - naive-rag        ：一问一检一答（只本地）
 * - rag-query-router ：路由后 complex 仍只本地检一轮
 * - rag-multihop     ：复杂题拆子问题，多轮本地检索
 * - 本文件           ：本地不够 → 博查联网补一次 → 再评估 → 生成
 *
 * 生活类比：
 * - 本地库   = 自家书架（Milvus ebook_collection，小说原文片段）
 * - 评估器   = 图书管理员：看桌上材料够不够答完这道题
 * - 联网搜索 = 去外面图书馆借补充资料（博查 Web Search，只允许借一次）
 * - generate = 拿「书架页 + 外借页」写最终答卷
 *
 * 为什么要联网兜底？
 * - 本地库只有小说文本，答不了「2013 版电视剧第几集」「给个可核对链接」这类题外信息
 * - 但也不能一上来就联网：浪费、也容易把小说事实搜歪；所以「本地优先、不够再补」
 *
 * 数据前提：
 * - Milvus 已有 ebook_collection（与 naive-rag 相同，只读不写）
 * - 环境变量 BOCHA_API_KEY（走 web_search 时需要）
 *
 * ----------------------------------------------------------------------------
 * 主流程怎么走？（对应 main + 图节点）
 * ----------------------------------------------------------------------------
 *  1. 定义 GraphState（问题、本地上下文、联网上下文、评估结果…）
 *  2. 装配图：
 *       START → route_question
 *         ├─ simple  → direct_answer → END
 *         └─ complex → local_retrieve → evaluate_local ─┬─ enough → generate → END
 *                                                       └─ 不够  → web_search → evaluate_local → generate → END
 *  3. main：连 Milvus → loadCollection → invoke
 *  4. 打印最终策略（流式回答已在 generate / direct_answer 里打出）
 *
 * 新手易混点：
 * - evaluate_local 会被跑最多两次：第一次只看本地；联网后回来再看「本地+联网」
 * - 一旦已有 webContext，条件边强制去 generate —— 防止「不够就再搜」死循环
 * - 检索结果不会自动进模型；generate 必须把 localContext / webContext 拼进 prompt
 * - 当前聊天网关对 withStructuredOutput(json_object) 不友好，本文件用「普通 invoke + 解析 JSON」
 *
 * 阅读建议：先扫导读与图装配（约文件后部 graph = …），再按
 * route → local_retrieve → evaluate → web_search → generate 顺序读各节点。
 * ============================================================================
 */

import 'dotenv/config';
import chalk from 'chalk';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { Milvus } from '@langchain/community/vectorstores/milvus';
/** model = 聊天模型；embeddings = 向量模型（见 ../model.mjs） */
import { model, embeddings } from '../model.mjs';

/** 各阶段控制台颜色：一眼区分路由 / 本地检索 / 评估 / 联网 / 生成 / 汇总 */
const log = {
  route: chalk.cyan,
  retrieve: chalk.yellow,
  evaluate: chalk.blue,
  web: chalk.magenta,
  answer: chalk.green,
  setup: chalk.gray,
  summary: chalk.whiteBright,
  err: chalk.red,
};

// ============================================================================
// 1. 图状态：整条流水线共用的「草稿纸」
// ============================================================================

/**
 * 字段怎么在节点间传（复杂路径）：
 * - question / k              ：用户原题与本地检索条数
 * - strategy / routeReason    ：路由结果（simple | complex）
 * - retrievedDocs             ：本地命中的结构化片段（带 score / 章节）
 * - localContext              ：把本地片段正文拼成一大段，给评估器 / 生成器用
 * - webContext                ：博查联网结果文本（空 = 还没搜过）
 * - evaluation                ：评估器输出的 JSON 字符串（enough / missing / web_query）
 * - generation                ：最终回答文本
 */
const GraphState = Annotation.Root({
  question: Annotation,
  k: Annotation,
  strategy: Annotation,
  routeReason: Annotation,
  retrievedDocs: Annotation,
  localContext: Annotation,
  webContext: Annotation,
  evaluation: Annotation,
  generation: Annotation,
});

/**
 * vectorStore：Milvus 客户端，模块级闭包供节点使用。
 * 必须在 main 里赋值并 load 后再 invoke，否则 local_retrieve 会空引用。
 */
let vectorStore;

// ============================================================================
// 2. 工具函数：解析 JSON、本地向量检索、博查联网
// ============================================================================

/**
 * 从模型回复里抽出 JSON 对象。
 * 为什么不用 withStructuredOutput：它会开 response_format=json_object；
 * 部分网关（如通义）还要求 messages 文本里出现英文单词 "json"，否则直接 400。
 * 所以改成「提示里写清要 JSON + 普通 invoke + 本函数抠出 {…}」。
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

/**
 * 对当前查询做向量相似度检索（本地书架抽页）。
 * 步骤：1. similaritySearchWithScore  2. 压成统一结构（含 score / 元数据）
 */
async function retrieveRelevantContent(query, k) {
  try {
    // 1. 问句 → 向量 → 在书架上找最近的 k 页（带分数）
    const docsWithScores = await vectorStore.similaritySearchWithScore(
      query,
      k
    );
    // 2. 统一字段，方便后面拼 localContext / 打日志
    return docsWithScores.map(([doc, score]) => ({
      score,
      content: doc.pageContent,
      id: doc.metadata?.id ?? 'unknown',
      book_id: doc.metadata?.book_id ?? '未知',
      chapter_num: doc.metadata?.chapter_num ?? '未知',
      index: doc.metadata?.index ?? '未知',
    }));
  } catch (error) {
    console.error(log.err('检索内容时出错:'), error.message);
    return [];
  }
}

/**
 * 调博查 Web Search，把网页结果压成可拼进 prompt 的纯文本。
 * 步骤：1. 校验 API Key  2. POST 搜索  3. 校验响应  4. 格式化引用列表
 * 环境变量：BOCHA_API_KEY
 */
async function bochaWebSearch(query, count) {
  // 1. 没 Key 就别调接口，尽早报错
  const apiKey = process.env.BOCHA_API_KEY;
  if (!apiKey) {
    throw new Error(
      'Bocha Web Search 的 API Key 未配置（环境变量 BOCHA_API_KEY）。'
    );
  }
  const url = 'https://api.bochaai.com/v1/web-search';
  const body = {
    query,
    freshness: 'noLimit',
    summary: true,
    count: count ?? 10,
  };

  // 2. 发请求（网络层失败单独提示）
  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw new Error(`搜索 API 请求失败（网络错误）：${error.message}`);
  }

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(
      `搜索 API 请求失败，状态码: ${response.status}, 错误信息: ${errorText}`
    );
  }

  // 3. 解析业务 JSON
  let json;
  try {
    json = await response.json();
  } catch (error) {
    throw new Error(`搜索结果解析失败：${error.message}`);
  }

  if (json?.code !== 200 || !json?.data) {
    throw new Error(`搜索 API 返回失败：${json?.msg ?? '未知错误'}`);
  }

  // 4. 无结果给一句占位；有结果则带上标题 / URL / 摘要，方便 generate 引用
  const webpages = json.data.webPages?.value ?? [];
  if (!webpages.length) {
    return '未找到相关结果。';
  }

  return webpages
    .map(
      (page, idx) => `引用: ${idx + 1}
标题: ${page.name}
URL: ${page.url}
摘要: ${page.summary}
网站名称: ${page.siteName}
网站图标: ${page.siteIcon}
发布时间: ${page.dateLastCrawled}`
    )
    .join('\n\n');
}

// ============================================================================
// 3. 节点：路由 —— 要不要走「本地检索 + 可能联网」？
// ============================================================================

/**
 * route_question：判断 simple（直答）还是 complex（本地检索链路）。
 * 步骤：1. 调模型拿 JSON  2. 解析 strategy  3. 清空检索相关状态
 */
const routeQuestionNode = async (state) => {
  console.log(log.route.bold('\n---ROUTE_QUESTION---'));
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
  console.log(log.route(`路由策略: ${strategy} (${reason})`));
  // 3. 清空上一轮残留，给复杂路径一个干净起点
  return {
    strategy,
    routeReason: reason,
    retrievedDocs: [],
    localContext: '',
    webContext: '',
    evaluation: '',
    generation: '',
  };
};

// ============================================================================
// 4. 节点：简单路径 —— 不检索，直接流式回答
// ============================================================================

/** simple 路径：不碰 Milvus / 博查，模型直接答 */
const directAnswerNode = async (state) => {
  console.log(log.answer.bold('\n---DIRECT_ANSWER---'));
  process.stdout.write(log.answer.bold('\n【AI 回答（流式）】\n'));
  let generation = '';
  const stream = await model.stream(`你是一个中文问答助手，请直接简洁回答问题。

问题：${state.question}
`);
  for await (const chunk of stream) {
    const text = typeof chunk.content === 'string' ? chunk.content : '';
    if (!text) continue;
    generation += text;
    process.stdout.write(log.answer(text));
  }
  process.stdout.write('\n');
  return { generation };
};

// ============================================================================
// 5. 节点：本地检索 —— 先去自家书架抽页
// ============================================================================

/**
 * local_retrieve：用原题做一次向量检索，写入 retrievedDocs + localContext。
 * 易混点：这里只「抽页上桌」，够不够答还要交给下一个评估节点。
 */
const retrieveLocalNode = async (state) => {
  console.log(log.retrieve.bold('\n---LOCAL_RETRIEVE---'));
  // 1. 向量检索
  const retrievedDocs = await retrieveRelevantContent(state.question, state.k);
  console.log(log.retrieve(`本地检索命中: ${retrievedDocs.length} 条`));
  // 2. 拼成纯文本上下文（评估器 / 生成器都读这个字段）
  const localContext = (retrievedDocs ?? []).map((d) => d.content).join('\n\n');
  return {
    retrievedDocs,
    localContext,
  };
};

// ============================================================================
// 6. 节点：评估 —— 桌上材料够不够？不够就开「外借条」
// ============================================================================

/**
 * evaluate_local：同一节点两种模式（看 webContext 是否已有内容）。
 * - 第一次（无 web）：只评本地；不够则产出 web_query 供联网用
 * - 第二次（有 web）：评「本地 + 联网」；之后条件边会强制去 generate
 * 步骤：1. 判断是否已联网  2. 调模型评估  3. 规范化字段  4. 写入 evaluation
 */
const evaluateNode = async (state) => {
  // 1. 有 webContext = 已经外借过一次，进入「二次评估」
  const hasWeb = Boolean(state.webContext && String(state.webContext).trim());
  console.log(
    log.evaluate.bold(
      hasWeb
        ? '\n---EVALUATE_CONTEXT_WITH_WEB---'
        : '\n---EVALUATE_LOCAL_CONTEXT---'
    )
  );
  // 2. 把现状摊给评估器；第一次才要求 web_query
  const response =
    await model.invoke(`你是信息充分性评估器。判断当前上下文是否足以回答用户问题，并用 JSON 返回结果。

用户问题：${state.question}

已检索上下文（来自本地知识库）：
${state.localContext || '（空）'}

${hasWeb ? `联网搜索结果：\n${state.webContext || '（空）'}\n` : ''}

请判断：
- enough: 是否足够回答（true/false）
- missing: 若不够，列出缺失信息点（最多 6 条字符串）
- reason: 简短原因
${hasWeb ? '' : '- web_query: 若不够，给出一个适合联网搜索的中文查询句（完整句，不用代词；够了可为空字符串）'}

只输出一行 JSON，格式严格为：{"enough":true|false,"missing":["..."],"reason":"...","web_query":"..."}`);

  // 3. 解析并钳制类型，避免模型胡写导致下游炸
  const parsed = parseJsonObject(response.content) ?? {};
  const out = {
    enough: parsed.enough === true,
    missing: Array.isArray(parsed.missing)
      ? parsed.missing.map((m) => String(m)).slice(0, 6)
      : [],
    reason: String(parsed.reason ?? response.content ?? ''),
    web_query: String(parsed.web_query ?? ''),
  };

  console.log(
    log.evaluate(
      `${hasWeb ? '二次评估' : '评估'}: enough=${out.enough} (${out.reason})`
    )
  );
  if (!out.enough && out.missing?.length) {
    out.missing.forEach((m, i) =>
      console.log(log.evaluate(`  缺失${i + 1}: ${m}`))
    );
  }
  // 4. 用字符串存进状态，方便条件边 / web_search 再 JSON.parse
  return {
    evaluation: JSON.stringify(out),
  };
};

// ============================================================================
// 7. 节点：联网 —— 按评估器开的「外借条」去搜
// ============================================================================

/**
 * web_search：读 evaluation.web_query（空则退回原题）→ 博查 → 写入 webContext。
 * 之后固定边回到 evaluate_local，做二次评估；不会在本节点直接生成答案。
 */
const webSearchNode = async (state) => {
  console.log(log.web.bold('\n---WEB_SEARCH---'));
  // 1. 取出评估器建议的联网查询句
  const parsed = (() => {
    try {
      return JSON.parse(state.evaluation || '{}');
    } catch {
      return {};
    }
  })();
  // 2. 没有合格 web_query 时，退回用户原题（总比空查询强）
  const query = (parsed.web_query ?? '').trim() || state.question;
  console.log(log.web(`联网查询: ${query}`));
  // 3. 调博查，结果整段塞进 webContext
  const webContext = await bochaWebSearch(query, 8);
  console.log(log.web(`联网结果长度: ${webContext.length}`));
  return { webContext };
};

// ============================================================================
// 8. 节点：生成 —— 本地 +（可选）联网上下文一起答
// ============================================================================

/**
 * generate：把 localContext 与 webContext 拼进 prompt 再答原题。
 * 易混点：答的是用户原始 question；检索/联网结果必须手动拼进上下文。
 */
const generateNode = async (state) => {
  console.log(log.answer.bold('\n---GENERATE---'));
  // 1. 有联网结果时用分隔线标出「外借补充」，方便模型区分来源
  const context = [state.localContext, state.webContext]
    .filter(Boolean)
    .join('\n\n===== 联网补充 =====\n\n');
  process.stdout.write(log.answer.bold('\n【AI 回答（流式）】\n'));
  let generation = '';
  // 2. 流式生成；要求不够就明说不确定，并尽量引用「引用: n / URL」
  const stream =
    await model.stream(`你是一个严谨的中文问答助手。优先依据上下文作答，不要编造。

上下文（本地知识库 + 可选联网补充）：
${context || '（空）'}

用户问题：${state.question}

回答要求：
1. 如果上下文足够，给出清晰、可核对的回答；需要时引用“引用: n / URL”或小说片段来支撑。
2. 如果上下文仍不足以确定关键事实，明确说明“不确定/无法从上下文确认”，并说明缺失点。
3. 不要输出表情符号。

回答：`);
  for await (const chunk of stream) {
    const text = typeof chunk.content === 'string' ? chunk.content : '';
    if (!text) continue;
    generation += text;
    process.stdout.write(log.answer(text));
  }
  process.stdout.write('\n');
  return { generation };
};

// ============================================================================
// 9. 条件边：路由后分支 / 评估后分支（防死循环的关键闸门）
// ============================================================================

/** 路由后：简单直答 vs 复杂走本地检索 */
function afterRoute(state) {
  return state.strategy === 'simple' ? 'direct_answer' : 'local_retrieve';
}

/**
 * 评估后下一步：
 * 1) 已有 webContext → 强制 generate（联网只允许一轮）
 * 2) 尚无联网且 enough → generate
 * 3) 尚无联网且不够 → web_search
 */
function afterEvaluateLocal(state) {
  // 1. 已经外借过 → 不再给第二次 web_search 机会
  if (state.webContext && String(state.webContext).trim()) {
    return 'generate';
  }
  // 2～3. 第一次评估：够就生成，不够才联网
  const parsed = (() => {
    try {
      return JSON.parse(state.evaluation || '{}');
    } catch {
      return {};
    }
  })();
  return parsed.enough === true ? 'generate' : 'web_search';
}

// ============================================================================
// 10. 装配图：把节点用边焊成可运行机器
// ============================================================================

/**
 * 拓扑（complex 主路径）：
 *   route → local_retrieve → evaluate ─┬─ generate → END
 *                                      └─ web_search → evaluate → generate → END
 * simple：route → direct_answer → END
 */
const graph = new StateGraph(GraphState)
  .addNode('route_question', routeQuestionNode)
  .addNode('direct_answer', directAnswerNode)
  .addNode('local_retrieve', retrieveLocalNode)
  .addNode('evaluate_local', evaluateNode)
  .addNode('web_search', webSearchNode)
  .addNode('generate', generateNode)
  .addEdge(START, 'route_question')
  .addConditionalEdges('route_question', afterRoute, {
    direct_answer: 'direct_answer',
    local_retrieve: 'local_retrieve',
  })
  .addEdge('local_retrieve', 'evaluate_local')
  .addConditionalEdges('evaluate_local', afterEvaluateLocal, {
    generate: 'generate',
    web_search: 'web_search',
  })
  // 联网后必须回到评估：二次评估 + 条件边强制 generate
  .addEdge('web_search', 'evaluate_local')
  .addEdge('direct_answer', END)
  .addEdge('generate', END)
  .compile();

// ============================================================================
// 11. 入口：连库 → 跑图 → 打印策略
// ============================================================================

async function main() {
  // 故意选「小说情节 + 电视剧集数/链接」题：前半本地可能够，后半通常要联网
  const question =
    '请回答《天龙八部》小说里“雁门关事件”的主谋是谁，并说明其儿子的最终结局；另外请补充：在《天龙八部》2013 版电视剧中，这段“雁门关事件”主要出现在哪几集？请给出可核对的来源链接。';
  const k = 8;

  // 11.1 导出 Mermaid，对照节点与「web → evaluate」回边
  const drawable = await graph.getGraphAsync();
  console.log(log.setup(drawable.drawMermaid({ withStyles: true })));

  // 11.2 连接已有集合（只读）
  console.log(log.setup('连接到 Milvus...'));
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
  console.log(log.setup('✓ 已连接\n'));

  // 11.3 检索前集合必须 load
  try {
    await vectorStore.client.loadCollection({
      collection_name: 'ebook_collection',
    });
    console.log(log.setup('✓ 集合 ebook_collection 已加载\n'));
  } catch (error) {
    if (!error.message.includes('already loaded')) throw error;
    console.log(log.setup('✓ 集合 ebook_collection 已处于加载状态\n'));
  }

  console.log(log.summary.bold('='.repeat(80)));
  console.log(log.summary.bold(`问题: ${question}`));
  console.log(log.summary.bold('='.repeat(80)));

  // 11.4 塞初始状态，跑完整条图（含可能的 web_search → evaluate 回环）
  const result = await graph.invoke({
    question,
    k,
    strategy: '',
    routeReason: '',
    retrievedDocs: [],
    localContext: '',
    webContext: '',
    evaluation: '',
    generation: '',
  });

  // 11.5 流式正文已在节点内打印；这里只回看策略
  console.log(log.summary.bold(`\n最终策略: ${result.strategy}`));
  if (!result.generation?.trim()) {
    console.log(log.err('模型未返回内容。'));
  }
}

/**
 * ----------------------------------------------------------------------------
 * 动手实验（可选）
 * ----------------------------------------------------------------------------
 * 1. 运行：node src/advanced-rag/rag-webfallback.mjs（需 BOCHA_API_KEY + Milvus）
 * 2. 看控制台是否出现 LOCAL_RETRIEVE → EVALUATE → WEB_SEARCH → EVALUATE → GENERATE
 * 3. 把 question 改成「什么是 RAG？」之类常识题，预期走 simple → DIRECT_ANSWER
 * 4. 把 question 改成纯小说情节且本地库能覆盖的题，预期评估 enough=true，跳过 WEB_SEARCH
 * ----------------------------------------------------------------------------
 */

main().catch((err) => {
  console.error(log.err('运行失败:'), err);
  process.exit(1);
});
