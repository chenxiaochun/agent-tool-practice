import 'dotenv/config';
import chalk from 'chalk';
import { pathToFileURL } from 'node:url';
/** Document：把纯文本收成 { pageContent, metadata }，供 compressDocuments 使用 */
import { Document } from '@langchain/core/documents';
/** BaseDocumentCompressor：重排器基类，约定实现 compressDocuments(docs, query) */
import { BaseDocumentCompressor } from '@langchain/core/retrievers/document_compressors';

export class DashscopeRerank extends BaseDocumentCompressor {
  constructor({ topN = 3 } = {}) {
    super();
    this.apiKey = process.env.OPENAI_RERANK_API_KEY;
    this.model = process.env.OPENAI_RERANK_MODEL_NAME;
    this.topN = topN;
    this.baseUrl = process.env.OPENAI_RERANK_BASE_URL;
  }

  async compressDocuments(documents, query, _callbacks) {
    const res = await fetch(this.baseUrl, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.model,
        input: {
          query,
          documents: documents.map((d) => d.pageContent),
        },
        parameters: {
          return_documents: false,
          top_n: this.topN,
        },
      }),
    });

    const json = await res.json();
    if (!res.ok) {
      throw new Error(
        `DashScope rerank ${res.status}: ${JSON.stringify(json)}`
      );
    }

    const results = json?.output?.results;
    if (!Array.isArray(results)) {
      throw new Error(`unexpected rerank response: ${JSON.stringify(json)}`);
    }

    return results.map((item) => documents[item.index]);
  }
}

async function main() {
  const compressor = new DashscopeRerank({ topN: 3 });

  const query = '什么是文本排序模型';
  const docs = [
    new Document({
      pageContent: '预训练语言模型的发展给文本排序模型带来了新的进展',
    }),
    new Document({
      pageContent: '量子计算是计算科学的一个前沿领域',
    }),
    new Document({
      pageContent: '文本排序模型广泛用于搜索引擎和推荐系统中…',
    }),
  ];

  const ranked = await compressor.compressDocuments(docs, query);
  console.log(chalk.whiteBright.bold('重排后顺序（pageContent）：'));
  for (const d of ranked) {
    console.log(chalk.green(`- ${d.pageContent}`));
  }
}

/** 仅直接运行本文件时才自测；被 hybrid-retrieval 等 import 时不触发 */
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
