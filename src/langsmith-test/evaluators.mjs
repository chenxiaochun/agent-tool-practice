/**
 * OpenEvals 内置 RAG 指标
 *
 * DashScope / 兼容网关：withStructuredOutput 会带 response_format=json_object，
 * 要求 messages 里出现 "json"，否则 400。OpenEvals 的 RAG prompt 本身不含该词，
 * 故用 system 补一句（createLLMAsJudge 支持 system，会拼到最前面）。
 */
import {
  createLLMAsJudge,
  RAG_GROUNDEDNESS_PROMPT,
  RAG_HELPFULNESS_PROMPT,
  RAG_RETRIEVAL_RELEVANCE_PROMPT,
} from 'openevals';
import { ChatOpenAI } from '@langchain/openai';

const judge = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL },
  model: process.env.MODEL_NAME ?? 'qwen-plus',
  temperature: 0,
});

/** 满足网关 json_object 校验；同时约束输出形态 */
const JUDGE_SYSTEM =
  'You must respond with a valid json object that contains the evaluation score (and reasoning when required).';

// RAG_GROUNDEDNESS_PROMPT —— 忠实度：答案是否被检索上下文支撑，有无幻觉
const ragGroundednessJudge = createLLMAsJudge({
  prompt: RAG_GROUNDEDNESS_PROMPT,
  feedbackKey: 'rag_groundedness',
  judge,
  system: JUDGE_SYSTEM,
  continuous: true,
});

// RAG_HELPFULNESS_PROMPT —— 回答有用性：是否切题、是否答非所问
const ragHelpfulnessJudge = createLLMAsJudge({
  prompt: RAG_HELPFULNESS_PROMPT,
  feedbackKey: 'rag_helpfulness',
  judge,
  system: JUDGE_SYSTEM,
  continuous: true,
});

// RAG_RETRIEVAL_RELEVANCE_PROMPT —— 检索相关性：召回片段与问题是否相关
const ragRetrievalRelevanceJudge = createLLMAsJudge({
  prompt: RAG_RETRIEVAL_RELEVANCE_PROMPT,
  feedbackKey: 'rag_retrieval_relevance',
  judge,
  system: JUDGE_SYSTEM,
  continuous: true,
});

export async function ragGroundednessEvaluator({ outputs }) {
  return ragGroundednessJudge({
    context: { documents: outputs.context },
    outputs: { answer: outputs.answer },
  });
}

export async function ragHelpfulnessEvaluator({ inputs, outputs }) {
  return ragHelpfulnessJudge({ inputs, outputs: { answer: outputs.answer } });
}

export async function ragRetrievalRelevanceEvaluator({ inputs, outputs }) {
  return ragRetrievalRelevanceJudge({
    inputs,
    context: { documents: outputs.context },
  });
}

export const ragEvaluators = [
  ragGroundednessEvaluator,
  ragHelpfulnessEvaluator,
  ragRetrievalRelevanceEvaluator,
];
