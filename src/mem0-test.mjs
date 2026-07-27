import 'dotenv/config'
import { MemoryClient } from 'mem0ai'

const USER_ID = 'demo-user'

function log(title, data) {
    console.log(`\n=== ${title} ===`)
    console.log(typeof data === 'string' ? data : JSON.stringify(data, null, 2))
}

async function main() {
    const client = new MemoryClient({
        apiKey: process.env.MEM0_API_KEY,
    })

    const conversation = [
        { role: 'user', content: '我是素食主义者，而且对坚果过敏。' },
        { role: 'assistant', content: '好的，我会记住你的饮食偏好。' },
        { role: 'user', content: '我住在北京，平时喜欢跑步。' },
        { role: 'assistant', content: '已记录：北京、爱好跑步。' },
    ]

    // 必须传 { userId }，至少提供 userId / agentId / appId / runId 之一
    const added = await client.add(conversation, { userId: USER_ID })
    log('Conversation added', added)

    const searchResult = await client.search('用户的饮食偏好和爱好是什么', {
        filters: { user_id: USER_ID },
        topK: 5,
    })
    log('Search result', searchResult)

    const allMemories = await client.getAll({
        filters: { user_id: USER_ID },
        pageSize: 10,
    })
    log('All memories', allMemories)

    // 如需清空该用户记忆，取消下面两行注释：
    const deleted = await client.deleteAll({ userId: USER_ID })
    log('Deleted', deleted)
}

main()
