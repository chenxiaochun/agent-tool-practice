import 'dotenv/config'
import { RunnableLambda, RunnablePassthrough, RunnableSequence, RunnableMap } from '@langchain/core/runnables'

const input = 'Hello World'

const chain = RunnableSequence.from([
    RunnableLambda.from((input) => ({ concept: input })),
    RunnableMap.from({
        original: new RunnablePassthrough(),
        processed: RunnableLambda.from((input) => {
            return {
                concept: input.concept,
                uppercase: input.concept.toUpperCase(),
                lowercase: input.concept.toLowerCase(),
            }
        }),
    })
])

const result = await chain.invoke(input)

console.log(result)