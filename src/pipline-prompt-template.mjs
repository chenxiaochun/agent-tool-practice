import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { PipelinePromptTemplate, PromptTemplate } from '@langchain/core/prompts';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
})

const personaPrompt = PromptTemplate.fromTemplate(`
   你是一位资深工程团队负责人，写作风格：{tone}    
   你擅长把复杂的技术问题转化为通俗易懂的文字，让非技术人员也能理解。
`)

const contextPrompt = PromptTemplate.fromTemplate(`
     公司名称：{company_name}
     部门：{team_name}
     汇报对象：{manager_name}
     本周时间范围：{week_range}
     本周部门核心目标：{team_goal}
`)

const taskPrompt = PromptTemplate.fromTemplate(`
    以下是本周团队的开发活动
    {dev_activities}
    请你从这些活动中提炼出：
    1. 本周团队的主要亮点
    2. 潜在风格和技术债
    3. 下周计划
`)

const formatPrompt = PromptTemplate.fromTemplate(`
    1. 本周概览（2-3 句话的 Summary）
    2. 详细拆分（按模块或项目分段）
    3. 关键指标表格，表头为：模块 | 亮点 | 风险 | 下周计划
    // 注意：
    - 尽量引用一些具体数据（如提交次数、完成的任务编号）
    - 语气专业，但可以偶尔带一点轻松的口吻，符合 {company_values}。
`)

const finalWeeklyPrompt = PromptTemplate.fromTemplate(`
    请根据以下内容生成一份完整的周报：

    ### 角色设定
    {persona}

    ### 背景信息
    {context}

    ### 任务要求
    {task}

    ### 输出格式
    {format}
`)

const pipelinePrompt = new PipelinePromptTemplate({
    pipelinePrompts: [
        { name: 'persona', prompt: personaPrompt },
        { name: 'context', prompt: contextPrompt },
        { name: 'task', prompt: taskPrompt },
        { name: 'format', prompt: formatPrompt },
    ],
    finalPrompt: finalWeeklyPrompt,
})

async function main() {
    const pipelineFormatted = await pipelinePrompt.format({
        tone: '专业但有人情味',
        company_name: '字节跳动',
        team_name: '字节跳动技术团队',
        manager_name: '张三',
        week_range: '2026-07-01 - 2026-07-07',
        team_goal: '完成字节跳动技术团队的目标',
        dev_activities: '完成字节跳动技术团队的目标,完成字节跳动技术团队的目标,完成字节跳动技术团队的目标',
        company_values: '字节跳动技术团队的核心价值观',
    })

    console.log(pipelineFormatted)
}

main()