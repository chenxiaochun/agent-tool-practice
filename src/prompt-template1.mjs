import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const naiveTemplate = PromptTemplate.fromTemplate(`
你是一名严谨但不失人情味的工程团队负责人，需要根据本周数据写一份周报。

公司名称：{company_name}

部门名称：{team_name}

直接汇报对象：{manager_name}

本周时间范围：{week_range}

本周团队核心目标：

{team_goal}

本周开发数据（Git提交/Jira任务）：

{dev_activities}

请根据以上信息生成一份【Markdown周报】，要求：

-有简短的整体summary（两三句话）

-有按模块/项目拆分的小结

-用一个Markdown表格列出关键指标（字段示例：模块/亮点/风险/下周计划）

-语气专业但有一点人情味，适合作为给老板和团队抄送的周报。
`)

async function main() {
    const prompt = await naiveTemplate.format({
        company_name: '公司名称',
        team_name: '部门名称',
        manager_name: '直接汇报对象',
        week_range: '本周时间范围',
        team_goal: '本周团队核心目标',
        dev_activities: '本周开发数据（Git提交/Jira任务）',
    });

    const response = await model.stream(prompt);

    for await (const chunk of response) {
        process.stdout.write(chunk.content);
    }
}

main()