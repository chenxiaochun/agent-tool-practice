import 'dotenv/config'
import { RunnableLambda, RunnableSequence } from '@langchain/core/runnables'

const addOne = RunnableLambda.from((x) => x + 1)

const multiplyByTwo = RunnableLambda.from((x) => x * 2)

const runnable = RunnableSequence.from([
    addOne,
    multiplyByTwo,
])

const result = await runnable.invoke(1)

console.log(result)