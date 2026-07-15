import 'dotenv/config'
import { RunnableMap, RunnableLambda, RunnableSequence } from '@langchain/core/runnables'
import { PromptTemplate } from '@langchain/core/prompts'

const addOne = RunnableLambda.from((input) => input.x + 1)

const multiplyByTwo = RunnableLambda.from((input) => input.x * 2)

const greetTemplate = PromptTemplate.fromTemplate('Hello, {name}!')

const weatherTemplate = PromptTemplate.fromTemplate('The weather in {city} is {weather}.')

const runnableMap = RunnableMap.from({
    addOne: addOne,
    multiplyByTwo: multiplyByTwo,
    greet: greetTemplate,
    weather: weatherTemplate,
})

const input = {
    name: 'John',
    city: 'New York',
    weather: 'sunny',
    x: 5,
}
const result = await runnableMap.invoke(input)

console.log(result)