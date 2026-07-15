import 'dotenv/config'
import { RouterRunnable, RunnableLambda } from '@langchain/core/runnables'

const toUppercase = RunnableLambda.from((input) => input.toUpperCase())

const toLowercase = RunnableLambda.from((input) => input.toLowerCase())

const router = new RouterRunnable({
    runnables: {
        uppercase: toUppercase,
        lowercase: toLowercase,
    },
})

const result1 = await router.invoke({
    key: 'uppercase',
    input: 'Hello, world!',
})

console.log(result1)

const result2 = await router.invoke({
    key: 'lowercase',
    input: 'Hello, world!',
})

console.log(result2)