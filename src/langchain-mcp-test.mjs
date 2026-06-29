import 'dotenv/config';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { ChatOpenAI } from '@langchain/openai';
import chalk from 'chalk';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';


const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const mcpClient = new MultiServerMCPClient({
    mcpServers: {
        'my-mcp-server': {
            command: 'node',
            args: ['/Users/chenxiaochun/Documents/MyProject/tool-test/src/my-mcp-server.mjs'],
        },
        "amap-maps-streamableHTTP": {
            url: 'https://mcp.amap.com/mcp?key=' + process.env.AMAP_API_KEY,
        },
        "filesystem": {
            command: 'npx',
            args: [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                ...(process.env.ALLOWED_PATHS.split(',') || '')
            ]
        },
        "chrome-devtools": {
            command: 'npx',
            args: [
                "-y",
                "chrome-devtools-mcp@latest",
            ]
        }
    },
})

const tools = await mcpClient.getTools();
const modelWithTools = model.bindTools(tools);

async function run(query, maxIterations = 30) {
    const messages = [
        new HumanMessage(query),
    ]

    for (let i = 0; i < maxIterations; i++) {
        console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));
        const response = await modelWithTools.invoke(messages);
        messages.push(response);

        if (!response.tool_calls || response.tool_calls.length === 0) {
            console.log(chalk.bgGreen(`✨ AI 最终回复:\n${response.content}\n`));
            return response.content;
        }

        console.log(chalk.bgBlue(`[检测到 ${response.tool_calls.length} 个工具调用]`));
        console.log(chalk.bgBlue(`[工具调用] ${response.tool_calls.map(call => `${call.name}(${JSON.stringify(call.args)})`).join('\n')}`));

        for (const toolCall of response.tool_calls) {
            const foundTool = tools.find(t => t.name === toolCall.name);
            if (foundTool) {
                const toolResult = await foundTool.invoke(toolCall.args);

                let contentStr = ''
                if (typeof toolResult === 'string') {
                    contentStr = toolResult;
                } else if (toolResult && toolResult.text) {
                    contentStr = toolResult.text;
                }

                messages.push(new ToolMessage({
                    content: contentStr,
                    tool_call_id: toolCall.id,
                }));
            }
        }

    }
    return messages[messages.length - 1].content;
}

const query = '北京南站附近的酒店， 并把酒店图片在浏览器中打开，每个酒店一个 tab 页';
const result = await run(query);
console.log(result);

await mcpClient.close();