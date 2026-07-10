import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate, FewShotChatMessagePromptTemplate } from '@langchain/core/prompts';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
})

const examplePrompt = ChatPromptTemplate.fromMessages([
    [
        'human',
        `用户输入：{user_requirement}期望周报结构：{expected_style}`,
    ],
    ['ai', '{report_snippet}'],
])

const examples = [
    {
        user_requirement:
            '重点突出稳定性治理，本周主要在修 Bug 和清理技术债，适合发给偏关注风险的老板。',
        expected_style: '语气稳健、偏保守，多强调风险识别和已做的兜底动作。',
        report_snippet:
            `- 支付链路本周共处理线上 P1 Bug 2 个、P2 Bug 3 个，全部在 SLA 内完成修复；\n` +
            `- 针对历史高频超时问题，完成 3 个核心接口的超时阈值和重试策略优化；\n` +
            `- 清理 12 条重复 / 噪音告警，减少值班同学 30% 的告警打扰。`,
    },
    {
        user_requirement:
            '偏向对外展示成果，希望多写一些亮点，适合发给更大范围的跨部门同学。',
        expected_style: '语气积极、突出成果，对技术细节做适度抽象。',
        report_snippet:
            `- 新上线「订单实时看板」，业务侧可以实时查看核心转化漏斗；\n` +
            `- 首次打通埋点 → 数据仓库 → 实时服务链路，为后续精细化运营提供基础能力；\n` +
            `- 和产品、运营一起完成 2 场内部分分享，会后收到 15 条正向反馈。`,
    },
]

const fewShotPrompt = new FewShotChatMessagePromptTemplate({
    examples,
    examplePrompt,
    inputVariables: [],
})

const finalPrompt = ChatPromptTemplate.fromMessages([
    [
        'system',
        '你是一名资深工程团队负责人。以下是一些周报撰写示例，请学习其风格与结构。',
    ],
    fewShotPrompt,
    [
        'human',
        `用户输入：{user_requirement}
期望周报结构：{expected_style}
请生成完整周报。`,
    ],
])

async function main() {
    const formattedPrompt = await finalPrompt.formatMessages({
        user_requirement:
            '本周主要在做 AI 周报工具，希望突出 Prompt 模块化设计和团队效率提升。',
        expected_style: '语气专业、结构清晰，适合发给技术负责人。',
    })
    console.log(formattedPrompt)

    const response = await model.invoke(formattedPrompt)
    console.log(response.content)
}

main()
