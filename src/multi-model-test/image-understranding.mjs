import 'dotenv/config';
import { multiModel } from '../model.mjs';
import { HumanMessage } from '@langchain/core/messages';

const response = await multiModel.invoke([
  new HumanMessage({
    content: [
      { type: 'text', text: '详细描述这张图片的内容' },
      {
        type: 'image_url',
        image_url: {
          url: 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg',
        },
      },
    ],
  }),
]);

console.log('model: ', process.env.MULTI_MODEL_NAME);
console.log(response.content);
