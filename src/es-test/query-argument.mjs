import { ChatPromptTemplate } from '@langchain/core/prompts';
import * as z from 'zod';

export const QueryAugmentationSchema = z.object({
  queries: z
    .array(z.string())
    .length(3)
    .describe(
      '恰好 3 条中文检索问句：不同角度改写或扩写；保留订单号、品牌等字面信息；不要编造事实'
    ),
});

const AUGMENT_PROMPT = ChatPromptTemplate.fromMessages([
  [
    'system',
    `用户会给出一句中文问题。请另外写出恰好 3 条检索用的问句（与原意一致、角度尽量不同），便于搜索引擎或向量库分别召回：
  可改写说法、换提问角度、或略加限定词；专有名词、型号、订单号等必须保留原样。
  必须互不相同，不要简单重复用户原句。
  请用 JSON 返回，只输出一行（花括号需写成双份以免被当成模板变量）：
  {{"queries":["问句1","问句2","问句3"]}}`,
  ],
  ['human', '{query}'],
]);

/** 从模型回复里抽出 JSON；兼容网关对 withStructuredOutput(json_object) 的限制 */
function parseJsonObject(raw) {
  const text = String(raw ?? '').trim();
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    return JSON.parse(match[0]);
  } catch {
    return null;
  }
}

/**
 * 清洗并凑满 3 条；不足时用 original 补齐。
 * 注意：若 list 为空，结果会是 [原句, 原句, 原句]——这就是终端里「三条完全一样」的来源。
 */
function normalizeThreeQueries(original, list) {
  const out = (list ?? [])
    .map((s) => (typeof s === 'string' ? s.trim() : ''))
    .filter(Boolean);
  while (out.length < 3) out.push(original);
  return out.slice(0, 3);
}

/**
 * 用聊天模型把用户问题扩成 3 条检索问句。
 *
 * 为什么不用 withStructuredOutput？
 * 当前部分网关要求 messages 里出现 "json" 才能用 response_format=json_object，
 * 否则会 400；旧实现把错误 catch 掉后走 normalizeThreeQueries(query, [])，
 * 于是三条全变成原句，看起来像「扩展没生效」。
 */
export async function augmentQuery(chatModel, query) {
  try {
    const msg = await AUGMENT_PROMPT.pipe(chatModel).invoke({ query });
    const content =
      typeof msg?.content === 'string'
        ? msg.content
        : Array.isArray(msg?.content)
          ? msg.content.map((c) => (typeof c === 'string' ? c : c?.text ?? '')).join('')
          : String(msg?.content ?? '');

    const parsed = parseJsonObject(content);
    const queries = normalizeThreeQueries(query, parsed?.queries);

    // 三条仍全等于原句 → 多半解析失败或模型偷懒，打个警告方便排查
    if (queries.every((q) => q === query)) {
      console.warn(
        '[query-argument] 扩展结果与原句相同；原始模型输出:',
        content.slice(0, 300)
      );
    }
    return { queries };
  } catch (err) {
    console.warn(
      '[query-argument] 查询扩展失败，回退为原句×3:',
      err?.message ?? err
    );
    return { queries: normalizeThreeQueries(query, []) };
  }
}

/** 原始问题在前，其后接 LLM 生成的问句；不做去重，顺序固定；每条各跑一次 ES、Milvus */
export function retrievalQueryStrings(original, augmentation) {
  return [original, ...(augmentation?.queries ?? [])]
    .map((s) => (typeof s === 'string' ? s.trim() : ''))
    .filter(Boolean);
}
