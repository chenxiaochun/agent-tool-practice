import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { XMLOutputParser } from '@langchain/core/output_parsers';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_APIKEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
})

const parser = new XMLOutputParser()
const query = `请提取以下文本中的人物信息：爱因斯坦出生于1879年3月14日，是一位伟大的物理学家。${parser.getFormatInstructions()}`

try {
    console.log('正在生成 XML 输出...')
    const response = await model.invoke(query)
    console.log(`模型原始咋就在：${response.content}`)

    const result = await parser.parse(response.content)
    console.log('解析后的结果：', result)
} catch (error) {
    console.error('Error generating XML output:', error);
}