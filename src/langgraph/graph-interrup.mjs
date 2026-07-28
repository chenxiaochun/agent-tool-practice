import { createInterface } from "node:readline/promises";
import {
    Annotation,
    Command,
    END,
    MemorySaver,
    START,
    StateGraph,
    interrupt,
} from "@langchain/langgraph";
import { model } from "../model.mjs";

const StateAnnotation = Annotation.Root({
    actionSummary: Annotation({
        reducer: (_prev, next) => next,
        default: () => "",
    }),
    userInput: Annotation({
        reducer: (_prev, next) => next,
        default: () => "",
    }),
    /** 模型判断：approve=肯定语气 / reject=否定语气 */
    verdict: Annotation({
        reducer: (_prev, next) => next,
        default: () => "",
    }),
    reason: Annotation({
        reducer: (_prev, next) => next,
        default: () => "",
    }),
    result: Annotation({
        reducer: (_prev, next) => next,
        default: () => "",
    }),
});

/** 从模型回复里抽出 JSON；失败则按关键词兜底 */
function parseVerdict(raw) {
    const text = String(raw ?? "").trim();
    const match = text.match(/\{[\s\S]*\}/);
    if (match) {
        try {
            const parsed = JSON.parse(match[0]);
            const verdict = parsed.verdict === "approve" ? "approve" : "reject";
            return {
                verdict,
                reason: String(parsed.reason ?? text),
            };
        } catch {
            // fall through
        }
    }
    const lower = text.toLowerCase();
    if (/\bapprove\b|肯定|同意|确认/.test(lower) && !/\breject\b|否定|拒绝/.test(lower)) {
        return { verdict: "approve", reason: text };
    }
    return { verdict: "reject", reason: text || "无法解析模型输出，默认拒绝" };
}

/** 展示一笔待确认的转账 */
const showTransfer = () => ({
    actionSummary: "向张三转账 ¥100（模拟，不会真扣款）",
});

/** 停在这里等人输入；resume 的值会写进 userInput */
const waitConfirm = (state) => {
    const text = interrupt({
        hint: "用自然语言确认或拒绝即可（如「好的」「不行」），模型会判断语气",
        actionSummary: state.actionSummary,
    });
    return { userInput: String(text).trim() };
};

/**
 * 用模型判断用户输入是肯定还是否定语气。
 * 注意：interrupt 之后再调模型，这样「等人」和「判语气」职责分开。
 */
const classify = async (state) => {
    const response = await model.invoke([
        {
            role: "system",
            content: [
                "你在判断用户对「转账确认」的语气。",
                "只要是肯定、同意、确认、可以、好的、嗯、行、没问题等语气 → verdict 填 approve。",
                "只要是否定、拒绝、取消、不要、算了、不行、不同意等语气 → verdict 填 reject。",
                "含糊不清时偏向 reject（安全优先）。",
                '只输出一行 JSON，格式严格为：{"verdict":"approve|reject","reason":"一句话理由"}',
            ].join("\n"),
        },
        {
            role: "user",
            content: `转账事项：${state.actionSummary}\n用户输入：${state.userInput}`,
        },
    ]);
    return parseVerdict(response.content);
};

/** 肯定语气 → 转账成功 */
const approve = (state) => ({
    result: `转账成功：${state.actionSummary}`,
});

/** 否定语气 → 转账失败 */
const reject = (state) => ({
    result: `转账失败：你的回复「${state.userInput}」被判定为否定语气，已取消本次转账`,
});

const graph = new StateGraph(StateAnnotation)
    .addNode("showTransfer", showTransfer)
    .addNode("waitConfirm", waitConfirm)
    .addNode("classify", classify)
    .addNode("approve", approve)
    .addNode("reject", reject)
    .addEdge(START, "showTransfer")
    .addEdge("showTransfer", "waitConfirm")
    .addEdge("waitConfirm", "classify")
    .addConditionalEdges("classify", (state) => state.verdict, {
        approve: "approve",
        reject: "reject",
    })
    .addEdge("approve", END)
    .addEdge("reject", END)
    .compile({ checkpointer: new MemorySaver() });

// 导出为 Mermaid：可复制到 https://mermaid.live 或 Markdown 的 ```mermaid 代码块
const drawable = await graph.getGraphAsync();
const mermaid = drawable.drawMermaid({ withStyles: true });
console.log(mermaid);

const config = { configurable: { thread_id: "interrupt-demo" } };

const paused = await graph.invoke({}, config);
console.log("\n待你确认：", paused.__interrupt__?.[0]?.value);

const rl = createInterface({ input: process.stdin, output: process.stdout });
const line = (await rl.question("> ")).trim();
await rl.close();

if (!line) {
    console.error("未输入，退出。");
    process.exit(1);
}

const done = await graph.invoke(new Command({ resume: line }), config);
console.log("判定：", done.verdict, "—", done.reason);
console.log("结果：", done.result);
console.log("完整状态：", done);
