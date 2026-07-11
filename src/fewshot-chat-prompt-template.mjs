import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { FewShotChatMessagePromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const EXAMPLES = [
    {
        input: '本周主要推进支付稳定性治理，做了事故处置、告警优化和演练。',
        output:
            '- 本周围绕支付链路稳定性开展治理工作：完成 1 起 P1 事故与 2 起 P2 事故的排查与修复，均在 SLA 内关闭；\n' +
            '- 梳理并合并冗余告警规则 8 条，新建 4 条基于 SLO 的告警，大幅降低无效告警噪音；\n' +
            '- 组织 1 次故障应急演练，验证支付核心链路的应急预案可行性。',
    },
    {
        input: '本周交付了新运营看板，并给业务同学做了多场分享。',
        output:
            '- 上线新一代「运营实时看板」，支持业务实时查看关键转化指标和漏斗数据；\n' +
            '- 衔接埋点、数据仓库与可视化链路，为后续精细化运营提供统一数据口径；\n' +
            '- 面向市场和运营团队组织 2 场产品培训，帮助非技术同学理解看板核心能力和使用场景。',
    },
];

const fewShotExamples = new FewShotChatMessagePromptTemplate({
    examplePrompt: ChatPromptTemplate.fromMessages([
        [
            'human',
            '本周主要推进支付稳定性治理，做了事故处置、告警优化和演练。',
        ],
        [
            'ai',
            '{output}'
        ]
    ]),
    exampleSeparator: '\n\n',
    examples: EXAMPLES,
    inputVariables: [],
})

const chatPrompt = ChatPromptTemplate.fromMessages([
    [
        'system',
        '你是一名资深技术负责人，请根据给定的工作内容，参考上面的示例，帮我写一段结构清晰、重点突出的周报片段（使用 Markdown 列表）。',
    ],
    [
        'system',
        '下面是若干参考示例，请重点学习它们的「表达方式和结构」，而不是照搬具体内容：',
    ],
    fewShotExamples,
    [
        'human',
        '这是我本周的实际工作内容，请帮我整理成周报：\n{current_work}',
    ],
]);

const currentWork =
    '本周完成了订单模块的一轮重构，拆分了历史遗留的大文件，并补齐了核心路径的单测；' +
    '同时修复了两起线上性能问题，并把指标接入统一监控看板。';

async function main() {
    const messages = await chatPrompt.formatMessages({
        current_work: currentWork,
    })

    try {
        const response = await model.stream(messages);
        for await (const chunk of response) {
            process.stdout.write(chunk.content);
        }
    } catch (error) {
        console.error('Error:', error);
        process.exit(0);
    }
}

main();