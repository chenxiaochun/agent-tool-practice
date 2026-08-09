import 'dotenv/config';
import { existsSync, mkdirSync } from 'node:fs';
import { createAgent, createMiddleware, HumanMessage } from 'langchain';
import {
  LocalShellBackend,
  createFilesystemMiddleware,
  createSkillsMiddleware,
} from 'deepagents';
import { model } from '../model.mjs';

const skills = '/.agents/skills/';
/** 虚拟路径（virtualMode）以 / 开头；对应磁盘上 rootDir 下的相对路径 */
const output = '/src/deepagents/output/deepagents-skills-flow.excalidraw';
const outputOnDisk = 'src/deepagents/output/deepagents-skills-flow.excalidraw';

if (!existsSync('.agents/skills/excalidraw-diagram-generator/SKILL.md')) {
  throw new Error(
    '未找到 excalidraw-diagram-generator，请先: npx skills add github/awesome-copilot --skill excalidraw-diagram-generator -y'
  );
}

mkdirSync('src/deepagents/output', { recursive: true });

const backend = await LocalShellBackend.create({
  rootDir: '.',
  virtualMode: true,
  inheritEnv: true,
});

/**
 * 模型写 .excalidraw 时，常把 content 传成 JSON 对象；
 * write_file schema 要求 content: string，否则 ToolInputParsingException。
 * 在真正 invoke 工具前把对象 JSON.stringify。
 */
const coerceWriteFileContent = createMiddleware({
  name: 'CoerceWriteFileContent',
  wrapToolCall: async (request, handler) => {
    const { toolCall } = request;
    if (toolCall?.name !== 'write_file') return handler(request);

    const args = toolCall.args ?? {};
    if (args.content == null || typeof args.content === 'string') {
      return handler(request);
    }

    const content =
      typeof args.content === 'object'
        ? JSON.stringify(args.content)
        : String(args.content);

    console.log(
      '[CoerceWriteFileContent] content 非字符串，已 JSON.stringify，长度:',
      content.length
    );

    return handler({
      ...request,
      toolCall: {
        ...toolCall,
        args: { ...args, content },
      },
    });
  },
});

const agent = createAgent({
  model,
  tools: [],
  systemPrompt: [
    '按 skills 库完成任务，需要时 read_file 对应 SKILL.md。中文回答。',
    '调用 write_file 时：content 必须是字符串；若写 JSON/.excalidraw，先序列化成 JSON 文本再传入，不要传对象。',
  ].join('\n'),
  middleware: [
    createSkillsMiddleware({ backend, sources: [skills] }),
    createFilesystemMiddleware({ backend }),
    coerceWriteFileContent,
  ],
});

const prompt = [
  '画一张流程图，描述本项目的 skills-agent 工作流：',
  '用户 Prompt → createAgent → createSkillsMiddleware → createFilesystemMiddleware → 模型回复。',
  `保存为 ${output}（write_file 的 content 必须是 JSON 字符串，不要传对象）。要求：`,
  '- 顶部大标题 + 副标题',
  '- 每个主节点 numbered（①②…）且框内 2～3 行中文说明',
  '- 右侧一列「说明：…」补充细节',
  '- 箭头上标注阶段名（如 invoke、wrapModelCall）',
  '- 底部图例（颜色含义 + 如何运行 demo）',
].join('\n');

console.log('用户:', prompt);

function chunkText(chunk) {
  if (!chunk?.content) return '';
  if (typeof chunk.content === 'string') return chunk.content;
  if (Array.isArray(chunk.content)) {
    return chunk.content
      .map((p) => (typeof p === 'string' ? p : (p?.text ?? '')))
      .join('');
  }
  return '';
}

const stream = await agent.streamEvents(
  { messages: [new HumanMessage(prompt)] },
  { recursionLimit: 100 }
);

let skillsMetadata;
console.log('\n--- 流式输出 ---\n');

try {
  for await (const event of stream) {
    if (event.event === 'on_chat_model_stream') {
      const text = chunkText(event.data?.chunk);
      if (text) process.stdout.write(text);
    }
    if (event.event === 'on_tool_start') {
      const name = event.name?.split('/').pop() ?? event.name;
      process.stdout.write(`\n\n→ ${name}\n\n`);
    }
    if (event.event === 'on_chain_end' && event.data?.output?.skillsMetadata) {
      skillsMetadata = event.data.output.skillsMetadata;
    }
  }
} catch (e) {
  console.error('\n\n[错误]', e.cause?.message ?? e.message);
  throw e;
}

console.log('\n');
console.log(
  'skills:',
  skillsMetadata?.map((s) => s.name)
);
if (existsSync(outputOnDisk)) {
  console.log('图表:', outputOnDisk);
  console.log('打开: https://excalidraw.com → Open → 选择该文件');
} else {
  console.log('未生成:', outputOnDisk);
}

await backend.close();
