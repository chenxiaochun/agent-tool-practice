/**
 * 图像编辑 — wan2.6-image
 * dashscope-sdk-official
 */
import 'dotenv/config';
import { writeFileSync } from 'node:fs';
import { Configuration, MultiModalConversation } from 'dashscope-sdk-official';

const imageUrl =
  'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg';

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
// 万相图像编辑走 DashScope 原生 multimodal-generation，不能用 ChatOpenAI
const client = new MultiModalConversation(configuration);

const result = await client.call({
  model: 'wan2.6-image',
  // 编辑任务：同一条 message 里同时传 { text } 指令和 { image } 原图 URL
  messages: [
    {
      role: 'user',
      content: [
        { text: '把图片背景改成下雪的冬天，人物保持不变' },
        { image: imageUrl },
      ],
    },
  ],
  prompt_extend: true, // 是否自动扩写提示词
  watermark: false, // 是否添加「AI 生成」水印
  n: 1, // 生成张数
  enable_interleave: false, // false = 图像编辑；true = 图文混排生成
  size: '1K', // 输出分辨率档位
});

if (result.status_code !== 200 || result.code) {
  throw new Error(result.message ?? `Request failed: ${result.status_code}`);
}

const resultUrl = result.output?.choices?.[0]?.message?.content?.[0]?.image;
if (!resultUrl) {
  throw new Error(`No image URL in response: ${JSON.stringify(result)}`);
}

console.log('model: wan2.6-image');
console.log('edited image URL:', resultUrl);

const imageResponse = await fetch(resultUrl);
writeFileSync(
  'output-wan-image-edit.png',
  Buffer.from(await imageResponse.arrayBuffer())
);
console.log('Saved to output-wan-image-edit.png');
