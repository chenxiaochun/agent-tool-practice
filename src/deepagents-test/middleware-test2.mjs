/**
 * ============================================================================
 * 新手导读：这份文件在干什么？
 * ============================================================================
 *
 * 演示 LangChain `createAgent` 的一种「扩展能力」写法：
 * **工具不写在 createAgent({ tools }) 里，而是由 middleware 挂上去**，
 * 并且用 `wrapToolCall` 在真正执行工具前后「包一层」——打日志、改返回文案、改 state。
 *
 * 可以把它想成安检通道：
 * - 模型决定要调某个工具（像旅客递证件）
 * - wrapToolCall = 安检员：先检查/记账，再放行真正执行，回来后还可盖章、改行李标签
 * - afterAgent = 下班清点：一共安检了几次
 *
 * 为什么工具要放进 middleware，而不是 createAgent 的 tools？
 * - createAgent 的 tools：Agent「出厂自带」的工具箱
 * - middleware.tools：插件额外塞进的工具（本文件 `tools: []`，时间工具全靠 middleware）
 * - wrapToolCall：所有经 Agent 发起的工具调用都会经过你，适合统一日志、限流、结果改写
 *
 * ----------------------------------------------------------------------------
 * 用户说「给我当前时间」之后，数据怎么流？
 * ----------------------------------------------------------------------------
 *  1. agent.invoke：带上 HumanMessage 进入图
 *  2. 模型看到 middleware 注册的 get_current_time，发起 tool_call
 *  3. wrapToolCall 拦截 → 打印「即将执行」→ handler(request) 真正跑工具
 *  4. 把原始 ToolMessage 改写成「内容 + 包装说明」，用 Command 写回 messages，
 *     并把 toolInvocationCount + 1
 *  5. 模型再读 ToolMessage，生成最终自然语言回复
 *  6. afterAgent：打印本轮累计工具调用次数
 *  7. 脚本打印最后一条消息内容，以及 state 里的 toolInvocationCount
 *
 * 阅读建议：先看 extendedToolsMiddleware 的 wrapToolCall，再看文末 for 循环入口。
 * 对照：middleware-test.mjs 侧重 before/afterModel；本文件侧重「工具从哪来 + 怎么包一层」。
 * ============================================================================
 */

import 'dotenv/config';
import { Command } from '@langchain/langgraph';
import { z } from 'zod';
import {
  createAgent,
  createMiddleware,
  HumanMessage,
  ToolMessage,
  tool,
} from 'langchain';
import { model } from '../model.mjs';

// ============================================================================
// 1. 定义一个极简工具：无参数，返回当前 UTC 时间字符串
// ============================================================================

/**
 * getCurrentTime：给模型用的「看表」工具。
 * schema 为空对象 {} → 调用时不需要任何参数。
 * 真正干活的是第一个函数参数：() => new Date().toISOString()
 */
const getCurrentTime = tool(() => new Date().toISOString(), {
  name: 'get_current_time',
  description: '返回当前 UTC 时间的 ISO 8601 字符串',
  schema: z.object({}),
});

// ============================================================================
// 2. Middleware：注册工具 + wrapToolCall 包装执行 + afterAgent 收尾统计
// ============================================================================

/**
 * ExtendedToolsMiddleware
 *
 * 做三件事：
 * 1) tools: [getCurrentTime] —— 把工具挂到 Agent 上（即使 createAgent 里 tools 是空的）
 * 2) wrapToolCall —— 每次工具执行的「洋葱皮」：前日志 → handler → 后改写结果 / 更新计数
 * 3) afterAgent —— 整轮 Agent 结束后打印 toolInvocationCount
 *
 * stateSchema.toolInvocationCount：middleware 自带的持久状态字段（本轮 invoke 内累加）。
 * 新手易混：改消息或计数时，这里用 Command({ update }) 回写；
 * 若直接 `return wrapped`（一个 ToolMessage），框架也能接，但就没法同时 bump 计数。
 */
const extendedToolsMiddleware = createMiddleware({
  name: 'ExtendedToolsMiddleware',
  stateSchema: z.object({
    /** 本轮 Agent 里工具被成功包装执行的次数；默认 0 */
    toolInvocationCount: z.number().default(0),
  }),
  /** 由 middleware 注入的工具列表（本 demo 的唯一工具来源） */
  tools: [getCurrentTime],

  /**
   * wrapToolCall：工具调用的外包装。
   *
   * 参数：
   * - request：本次工具调用请求（含 tool / toolCall / state 等）
   * - handler：框架提供的「真正执行工具」函数；你必须在适当时机 await handler(request)
   *
   * 步骤：
   * 1. 解析工具名，打「即将执行」日志
   * 2. await handler(request) 拿到结果
   * 3. 若不是 ToolMessage（例如别处已返回 Command），原样放行
   * 4. 用新的 ToolMessage 包一层说明文字（演示「可改写工具输出」）
   * 5. 用 Command.update 同时：messages 追加包装后的结果 + toolInvocationCount+1
   */
  wrapToolCall: async (request, handler) => {
    // 1. 工具名：优先用绑定好的 tool.name，否则用 toolCall.name
    const toolName = request.tool?.name ?? request.toolCall.name;
    console.log(
      `[Tools] 即将执行: ${toolName}`,
      'args:',
      request.toolCall.args ?? {}
    );

    // 2. 放行：真正调用 getCurrentTime（或其它工具）
    const result = await handler(request);

    // 3. 非 ToolMessage 不包装（避免误伤高级控制流）
    if (!ToolMessage.isInstance(result)) return result;

    // 4. 改写内容：原输出 + 一行「已被 middleware 包装」的标记（方便你在终端辨认）
    const wrapped = new ToolMessage({
      content: `${result.content}\n[wrapToolCall] 已由 ExtendedToolsMiddleware 包装`,
      tool_call_id: result.tool_call_id, // 必须原样带回，模型才能把结果对上某次 tool_call
      name: result.name,
    });
    console.log(
      `[Tools] 执行完成: ${toolName}`,
      typeof wrapped.content === 'string'
        ? wrapped.content.slice(0, 120)
        : wrapped
    );

    // 5. Command：一次更新多个 state 字段（消息列表 + 计数）
    //    新手易混：messages: [wrapped] 在这里表示「追加这条 ToolMessage」，
    //    不是用数组整体替换整段历史（由 Agent/图的 reducer 合并）。
    return new Command({
      update: {
        toolInvocationCount: request.state.toolInvocationCount + 1,
        messages: [wrapped],
      },
    });
  },

  /** Agent 整轮结束（含可能的多轮 tool↔model）后调用，做收尾日志 */
  afterAgent: (state) => {
    console.log(
      `[Tools] agent 结束，middleware 统计工具调用: ${state.toolInvocationCount} 次`
    );
  },
});

// ============================================================================
// 3. 装配 Agent：模型有了，但 tools 故意留空——能力来自上面的 middleware
// ============================================================================

/**
 * tools: [] —— 刻意不在这里注册 getCurrentTime。
 * 若运行后模型仍能「查时间」，说明 middleware.tools 生效了。
 */
const agent = createAgent({
  model,
  tools: [],
  systemPrompt: '你是一个助手。',
  middleware: [extendedToolsMiddleware],
});

// ============================================================================
// 4. 入口：发一句会触发「查时间」的用户话，观察日志与计数
// ============================================================================

for (const text of ['给我当前时间']) {
  console.log('\n用户:', text);
  // invoke 返回的 state 里会带上 middleware 声明的 toolInvocationCount
  const { messages, toolInvocationCount } = await agent.invoke({
    messages: [new HumanMessage(text)],
  });
  console.log('回复:', messages.at(-1)?.content);
  console.log('toolInvocationCount:', toolInvocationCount);
}

/**
 * ----------------------------------------------------------------------------
 * 动手实验（可选）
 * ----------------------------------------------------------------------------
 * 运行：node src/deepagents-test/middleware-test2.mjs
 *
 * 预期终端大致出现：
 * 1. [Tools] 即将执行: get_current_time
 * 2. [Tools] 执行完成: … 内容里含 ISO 时间 +「已由 ExtendedToolsMiddleware 包装」
 * 3. [Tools] agent 结束，middleware 统计工具调用: 1 次（若模型只调了一次）
 * 4. 最终「回复」是自然语言；toolInvocationCount 与上面统计一致
 *
 * 对比实验：
 * - 把 middleware 里的 tools: [getCurrentTime] 改成 tools: []，
 *   模型通常会说没法查时间 / 瞎编——证明工具来自 middleware。
 * - 把 wrapToolCall 里 return new Command(...) 改成直接 return wrapped，
 *   时间仍能回，但 toolInvocationCount 往往一直是 0（没人 +1）。
 * ============================================================================
 */
