/**
 * 图生视频 — wan2.6-i2v-flash
 * dashscope-sdk-official（自动轮询异步任务）
 */
import 'dotenv/config';
import { writeFileSync } from 'node:fs';
import { Configuration, VideoSynthesis } from 'dashscope-sdk-official';

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
// 视频生成是异步任务；VideoSynthesis.call 内部会提交任务并轮询至完成
const client = new VideoSynthesis(configuration);

console.log('model: wan2.6-i2v-flash');
console.log('creating video task...');

const result = await client.call({
  model: 'wan2.6-i2v-flash',
  prompt: '女孩缓缓转头，海风吹动头发，阳光洒在沙滩上，镜头缓慢推进', // 运动/镜头描述
  img_url:
    'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg', // 首帧参考图，图生视频必填
  resolution: '720P', // 图生视频用 resolution（如 720P / 1080P）
  prompt_extend: true, // 是否自动扩写提示词
  duration: 5, // 视频时长（秒）
});

const taskStatus = result.output?.task_status;
console.log('task_status:', taskStatus);

if (taskStatus === 'FAILED') {
  throw new Error(result.output?.message ?? result.message ?? 'Task failed');
}

const videoUrl = result.output?.video_url;
if (!videoUrl) {
  throw new Error(`No video URL in response: ${JSON.stringify(result)}`);
}

console.log('video URL:', videoUrl);
const videoResponse = await fetch(videoUrl);
writeFileSync(
  'output-wan-image-to-video.mp4',
  Buffer.from(await videoResponse.arrayBuffer())
);
console.log('Saved to output-wan-image-to-video.mp4');
