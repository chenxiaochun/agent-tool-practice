import { Annotation, END, START, StateGraph } from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
    tries: Annotation({
        reducer: (_prev, next) => next,
        default: () => 0,
    }),
    ok: Annotation({
        reducer: (_prev, next) => next,
        default: () => false,
    }),
    message: Annotation({
        reducer: (prev, next) => prev.concat(next),
        default: () => "",
    }),
});

const attempt = (state) => {
    const tries = state.tries + 1;
    const ok = tries >= 3;
    return {
        tries,
        ok,
        message: ok ? `第 ${tries} 次成功` : `第 ${tries} 次失败，继续重试`,
    };
};

const graph = new StateGraph(StateAnnotation)
    .addNode("attempt", attempt)
    .addEdge(START, "attempt")
    /**
     * addConditionalEdges：从「attempt」出来时不走死边，而是按状态选下一条路。
     *
     * 三个参数分别是：
     *  1. 源节点名 "attempt" —— 这个节点跑完后，才做路由判断
     *  2. 路由函数 (state) => ... —— 读当前草稿纸，返回一个「路径标签」
     *     - state.ok === true  → 返回 "done"（成功，该收工）
     *     - state.ok === false → 返回 "retry"（失败，再试一次）
     *  3. 路径表 { 标签: 目标 } —— 把标签翻译成真正要去的节点 / END
     *     - "retry" → 再进 "attempt"（形成循环，像「失败就重跑同一关」）
     *     - "done"  → END（走出图，invoke 结束）
     *
     * 本例实际跑法（tries 从 0 起步）：
     *   第 1 次 attempt → ok=false → retry → 回到 attempt
     *   第 2 次 attempt → ok=false → retry → 回到 attempt
     *   第 3 次 attempt → ok=true  → done  → END
     *
     * 新手易混点：
     * - 路由函数返回的是「标签」（path map 的 key），不是节点函数本身
     * - 和固定 addEdge 不同：这里边是「运行时才定」，所以能做出重试环
     * - 若没有 "retry" → "attempt" 这条自环，失败后就无处可去
     */
    .addConditionalEdges("attempt", (state) => (state.ok ? "done" : "retry"), {
        retry: "attempt", // 失败：标签 retry → 再跑 attempt（循环）
        done: END, // 成功：标签 done → 结束
    })
    .compile();

// 导出为 Mermaid：可复制到 https://mermaid.live 或 Markdown 的 ```mermaid 代码块
const drawable = await graph.getGraphAsync();
const mermaid = drawable.drawMermaid({ withStyles: true });
console.log(mermaid);

const result = await graph.invoke({ tries: 0 });
console.log("result:", result);