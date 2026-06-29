import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const database = {
    users: {
        "001": {
            id: "001",
            name: "John Doe",
            email: "john.doe@example.com",
            age: 30,
            address: "123 Main St, Anytown, USA",
            phone: "123-456-7890",
        },
        "002": {
            id: "002",
            name: "Jane Smith",
            email: "jane.smith@example.com",
            age: 25,
            address: "456 Main St, Anytown, USA",
            phone: "123-456-7890",
        },
        "003": {
            id: "003",
            name: "Jim Beam",
            email: "jim.beam@example.com",
            age: 35,
            address: "789 Main St, Anytown, USA",
            phone: "123-456-7890",
        },
        "004": {
        }
    }
}

const server = new McpServer({
    name: "my-mcp-server",
    version: "1.0.0",
})

server.registerTool('query-tool', {
    description: "查询数据库中的用户信息，输入用户id，返回用户信息",
    inputSchema: z.object({
        userId: z.string().describe("用户id, 例如: 001，002，003，004"),
    }),
}, async ({ userId }) => {
    const user = database.users[userId];
    if (!user) {
        return {
            content: [
                {
                    type: "text",
                    text: `用户不存在: ${userId}`
                }
            ]
        }
    }
    return {
        content: [
            {
                type: "text",
                text: `用户信息: ${JSON.stringify(user)}`
            }
        ]
    }
});

server.registerResource('使用指南', 'docs://guide', {
    description: "使用指南",
    mimeType: "text/plain",
}, async () => {
    return {
        contents: [
            {
                uri: 'docs://guide',
                mimeType: "text/plain",
                text: '使用指南: 输入用户id，返回用户信息'
            }
        ]
    }
})

const transport = new StdioServerTransport();
await server.connect(transport);