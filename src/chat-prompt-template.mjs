import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

async function main() {
    const chatPrompt = ChatPromptTemplate.fromMessages([
        [
            'system',
            '你是一位资深工程团队负责人，擅长用结构化易读的方式写技术周报，写作风格要求：{tone}'
        ],
        [
            'human',
            `本周信息如下，
        公司名称: {company_name}
        团队名称: {team_name}
        直接汇报对象: {manager_name}
        本周时间范围: {week_range}
        本周团队核心目标:
                {team_goal}
        本周开发数据(Git 提交 / Jira 任务等):
                {dev_activities}
        请据此输出一份 Markdown 周报，结构建议包含：
        1. 本周概览(2 - 3 句话)
        2. 详细拆分(按项目或模块分段)
        3. 关键指标表格(字段示例: 模块 / 亮点 / 风险 / 下周计划)
        语气专业但有人情味。`
        ]
    ])

    const chatMessages = await chatPrompt.formatMessages({
        company_name: '字节跳动',
        team_name: '字节跳动技术团队',
        manager_name: '张三',
        week_range: '2026-07-01 - 2026-07-07',
        team_goal: '完成字节跳动技术团队的目标',
        dev_activities: '完成字节跳动技术团队的目标',
        tone: '专业但有人情味'
    })

    console.log(chatMessages)
    const response = await model.invoke(chatMessages)
    console.log(response.content)
}

main()