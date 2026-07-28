import "dotenv/config";

import { HumanMessage } from "@langchain/core/messages";
import { createSupervisor } from "@langchain/langgraph-supervisor";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent, tool } from "langchain";
import { z } from "zod";

import { lookupCityTrivia, lookupWeather } from "./simple-mock.mjs";

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const lookupWeatherTool = tool(
    async ({ city }) => lookupWeather(city),
    {
        name: "lookup_weather",
        description: "查询某城市当日天气概况（气温区间、天气、空气质量等）。",
        schema: z.object({
            city: z.string().describe("城市名，如 杭州"),
        }),
    }
);

const lookupCityTriviaTool = tool(
    async ({ city }) => lookupCityTrivia(city),
    {
        name: "lookup_city_trivia",
        description: "查询与某城市相关的一句趣味知识。",
        schema: z.object({
            city: z.string().describe("城市名，如 杭州"),
        }),
    }
);

/** 子代理 A：只回答「天气」类问题 */
const weatherAgent = createAgent({
    name: "weather_agent",
    description: "专门查天气",
    model,
    tools: [lookupWeatherTool],
    systemPrompt: "你只处理天气。用户提到城市时，用 lookup_weather 查询后再用中文简短说明。",
});

/** 子代理 B：只回答「城市小知识」 */
const triviaAgent = createAgent({
    name: "trivia_agent",
    description: "专门讲与城市相关的小知识；必须调用 lookup_city_trivia。",
    model,
    tools: [lookupCityTriviaTool],
    systemPrompt: "你只讲城市小知识。先 lookup_city_trivia，再用人话转述，不要编造工具里没有的内容。",
});

/**
 * Supervisor：根据用户问的是「天气」还是「小知识」切换子代理。
 * （真实业务里还可以再加更多子代理，思路一样。）
 */
const workflow = createSupervisor({
    agents: [weatherAgent.graph, triviaAgent.graph],
    llm: model,
    prompt: `
        你是调度员，只负责选人，不要自己报气温、也不要自己讲城市百科。
        - 问天气、气温、下不下雨、空气 → 用 weather_agent
        - 问小知识、名胜、历史、一句介绍 → 用 trivia_agent
    `,
});

/**
 * compile：把「设计好的 supervisor 工作流」焊成可运行的 CompiledGraph。
 * 类比：前面 createSupervisor 是画组织架构图，compile 之后才真正能派活。
 */
const app = workflow.compile();

// ============================================================================
// 可视化：导出 Mermaid，方便对照「调度员 ↔ 子代理」怎么连
// ============================================================================

const drawable = await app.getGraphAsync();
console.log(drawable.drawMermaid({ withStyles: true }));

// ============================================================================
// 准备输入：多代理图通常吃 messages 列表（对话草稿纸）
// ============================================================================

/**
 * input.messages：本轮用户话。
 * 故意一句话里同时要「天气 + 小知识」，好观察 supervisor 是否先后调度两个子代理。
 */
const input = {
    messages: [
        new HumanMessage("查一下杭州的天气，再讲一条和杭州有关的小知识。"),
    ],
};

// ============================================================================
// 流式跑图：边跑边记「走过哪些节点」+「最终整份状态」
// ============================================================================

/** nodePath：按出现顺序记下节点名，用来打印「调度轨迹」 */
const nodePath = [];
/** finalState：最后一次 values 事件里的完整状态（含全部 messages） */
let finalState = null;

/**
 * stream vs invoke：
 * - invoke：闷头跑完，只拿最终结果
 * - stream：每走一步就推送事件，适合看「谁被叫到了」
 *
 * streamMode 同时开两种：
 * - "updates"：只含「本步改动了什么」，payload 形如 { 节点名: 更新内容 }
 *   → 用 Object.keys(payload) 就能知道刚跑完哪个节点
 * - "values"：推送「合并后的完整状态快照」
 *   → 不断刷新 finalState，循环结束后就是终态
 *
 * 新手易混点：
 * - 一次 for-await 可能收到两种 mode 的事件，所以要看 event[0]
 * - updates 里的 key 是节点名，不是消息内容；消息在 values 的 messages 里
 */
const stream = await app.stream(input, { streamMode: ["updates", "values"] });
for await (const event of stream) {
    // event ≈ ["updates" | "values", payload]
    const [mode, payload] = event;
    if (mode === "updates" && payload && typeof payload === "object") {
        // 本步跑过的节点名追加进路径（一个 updates 里通常一个 key）
        nodePath.push(...Object.keys(payload));
    } else if (mode === "values") {
        // 用最新快照覆盖；循环结束时即为最终 state
        finalState = payload;
    }
}

console.log("路径:", nodePath.join(" → "));
const last = finalState?.messages?.at(-1);
console.log(last?.content ?? finalState?.messages);