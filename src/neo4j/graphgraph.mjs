import 'dotenv/config';
import chalk from 'chalk';
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph';
import { ChatOpenAI } from '@langchain/openai';
import { StateGraph, END, START } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';

/** 各阶段控制台颜色：问题 / Cypher / 检索 / 回答 / 结构图 */
const log = {
  setup: chalk.gray,
  question: chalk.cyan,
  cypher: chalk.yellow,
  context: chalk.blue,
  answer: chalk.green,
  divider: chalk.whiteBright,
  err: chalk.red,
};

// ----------------------
// 连接 Neo4j 知识图谱
// ----------------------
const graph = new Neo4jGraph({
  url: 'bolt://localhost:7687',
  username: 'neo4j',
  password: '12345678',
});

// ----------------------
// 大模型
// ----------------------
const llm = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  temperature: 0,
  configuration: { baseURL: process.env.OPENAI_BASE_URL },
});

// ----------------------
// 定义状态
// ----------------------
const state = {
  messages: {
    value: (left, right) => left.concat(Array.isArray(right) ? right : [right]),
    default: () => [],
  },
  cypher: null,
  context: null,
  answer: null,
};

function userQuery(state) {
  const last = state.messages[state.messages.length - 1];
  return last.content;
}

// ----------------------
// 步骤1：生成 Cypher
// ----------------------
async function generateCypher(state) {
  const prompt = `
      你是一个专业的 Neo4j Cypher 生成器。
      严格按照下面的结构生成正确语句，只返回纯 Cypher 代码，不要任何解释、不要标点、不要 markdown。
  
      节点：
      - Product: 奶茶产品
      - Ingredient: 配料
      - Type: 奶茶类型
      - Method: 制作工艺
      - People: 适合人群
  
      属性（所有节点统一，禁止用中文属性名）：
      - 一律使用 name（英文），例如 Product {name: '珍珠奶茶'}
      - 禁止 名称 / 品名 / title 等其它键；写错会导致 MATCH 结果为空
  
      关系方向（必须严格遵守）：
      - (Product)-[:属于]->(Type)
      - (Product)-[:包含]->(Ingredient)
      - (Product)-[:适合]->(People)
      - (Ingredient)-[:使用]->(Method)
  
      规则：
      1. 关系方向绝对不能反
      2. 多跳查询请使用多个 MATCH，不要把无关边串成一条路径
      3. 只返回最终可运行的 Cypher 语句
      4. 过滤条件必须写成 {name: '...'}，不要写成 {名称: '...'}
      5. 「包含」只存在于 Product→Ingredient；「适合」只存在于 Product→People
         ——禁止写出 (Type)-[:包含]->(Ingredient) 这类链式路径
      6. 问某 Type 下饮品的配料/人群时，正确模式示例：
         MATCH (p:Product)-[:属于]->(t:Type {name: '台式奶茶'})
         MATCH (p)-[:包含]->(i:Ingredient)
         RETURN i.name
  
      用户问题：${userQuery(state)}
    `;
  const res = await llm.invoke([new HumanMessage(prompt)]);
  return { cypher: res.content };
}

// ----------------------
// 步骤2：执行图查询
// ----------------------
async function executeGraphQuery(state) {
  try {
    const res = await graph.query(state.cypher);
    return { context: JSON.stringify(res) };
  } catch (e) {
    return { context: '未查询到相关知识' };
  }
}

// ----------------------
// 步骤3：生成答案
// ----------------------
async function generateAnswer(state) {
  const prompt = `
    你是奶茶专家，根据下方「检索结果」回答用户问题；检索结果为空或不足时简要说明无法从图谱得到答案，不要编造。
    回答要求：
    - 直接列出事实，不要推断图谱里未出现的配料（如水、冰、添加剂等）。

    检索结果：${state.context}
    用户问题：${userQuery(state)}
  `;
  const res = await llm.invoke([new HumanMessage(prompt)]);
  return { answer: res.content };
}

// ----------------------
// 构建 LangGraph 工作流
// ----------------------
const workflow = new StateGraph({ channels: state })
  .addNode('generateCypher', generateCypher)
  .addNode('executeGraph', executeGraphQuery)
  .addNode('generateAnswer', generateAnswer)
  .addEdge(START, 'generateCypher')
  .addEdge('generateCypher', 'executeGraph')
  .addEdge('executeGraph', 'generateAnswer')
  .addEdge('generateAnswer', END);

const app = workflow.compile();

async function printWorkflowMermaid() {
  const drawable = await app.getGraphAsync();
  const mermaid = drawable.drawMermaid({ withStyles: true });
  console.log(log.setup.bold('--- LangGraph 工作流 (Mermaid) ---'));
  console.log(log.setup(mermaid));
  console.log(
    log.setup('-----------------------------------------------------------')
  );
}

// ----------------------
// 运行 GraphRAG
// ----------------------
async function runGraphRAG(question) {
  const res = await app.invoke({
    messages: [new HumanMessage(question)],
  });

  console.log(log.divider.bold('======================================'));
  console.log(log.question.bold('用户问题：'), log.question(question));
  console.log(log.cypher.bold('生成 Cypher：'), log.cypher(res.cypher ?? ''));
  console.log(
    log.context.bold('检索结果：'),
    log.context(
      typeof res.context === 'string'
        ? res.context
        : JSON.stringify(res.context)
    )
  );
  console.log(
    log.answer.bold('最终回答：'),
    log.answer(res.answer ?? log.err('（空）'))
  );
  console.log(log.divider.bold('======================================'));
}

// ======================
// 测试
// ======================
(async () => {
  await printWorkflowMermaid();
  await Promise.all([
    runGraphRAG('我们这款珍珠奶茶有哪些配料？'),
    runGraphRAG('台式奶茶的饮品都有哪些配料？'),
    runGraphRAG('珍珠奶茶适合哪些人群饮用？'),
  ]);
  process.exit(0);
})().catch(console.error);
