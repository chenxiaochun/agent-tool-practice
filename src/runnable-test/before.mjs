import 'dotenv/config'
import { StructuredOutputParser } from '@langchain/core/output_parsers'
import { PromptTemplate } from '@langchain/core/prompts'
import { z } from 'zod'
import { ChatOpenAI } from '@langchain/openai'

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
})

const schema = z.object({
    translation: z.string().describe('The translation of the text'),
    keywords: z.array(z.string()).describe('The keywords of the text'),
})

const outputParser = StructuredOutputParser.fromZodSchema(schema)

const promptTemplate = PromptTemplate.fromTemplate(
    '将以下文本翻译成英文，并提取关键词。文本：{text}\n\n输出格式：{format_instructions}'
)

const input = {
    text: '你好，世界！',
    format_instructions: outputParser.getFormatInstructions(),
}

async function main() {
    const formattedPrompt = await promptTemplate.format(input)

    const response = await model.invoke(formattedPrompt)

    const result = await outputParser.parse(response.content)

    console.log(result)
}

main()