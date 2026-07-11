import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import mysql from 'mysql2/promise';

const model = new ChatOpenAI({
    modelName: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    temperature: 0,
});

const friendSchema = z.object({
    name: z.string().describe('The name of the friend'),
    gender: z.enum(['male', 'female']).describe('The gender of the friend'),
    birthday: z.string().describe('The birthday of the friend'),
    company: z.string().describe('The company of the friend'),
    title: z.string().describe('The title of the friend'),
    phone: z.string().describe('The phone of the friend'),
    wechat: z.string().describe('The wechat of the friend'),
});

const friendsArraySchema = z.array(friendSchema).describe('The array of friends');

const structuredModel = model.withStructuredOutput(friendsArraySchema);

const connectionConfig = {
    host: process.env.MYSQL_HOST,
    port: Number(process.env.MYSQL_PORT ?? 3306),
    user: process.env.MYSQL_USER,
    password: process.env.MYSQL_PASSWORD,
    database: process.env.MYSQL_DATABASE,
    multipleStatements: true,
}

async function extractAndInsert(text) {
    const connection = await mysql.createConnection(connectionConfig);
    try {
        await connection.query('USE hello');

        console.log(`正在从文本中提取好友信息...`);
        const prompt = `
            请从以下信息中提取所有好友信息，文本中可能包含一个或者多个人的信息，请将每个人的信息提取出来，并返回一个数组，数组中每个元素是一个对象，对象中包含好友的姓名、性别、生日、公司、职位、电话、微信。
            ${text}
            
        `
        const response = await structuredModel.invoke(prompt);
        console.log(`提取到的好友信息:`, response);

        const insertSql = `
            INSERT INTO friends (name, gender, birthday, company, title, phone, wechat) VALUES ?
        `
        const rows = response.map((friend) => [
            friend.name,
            friend.gender,
            friend.birthday,
            friend.company,
            friend.title,
            friend.phone,
            friend.wechat,
        ])

        const [result] = await connection.query(insertSql, [rows]);
        console.log(`插入成功，插入 ${result.affectedRows} 条数据`);
    } catch (error) {
        console.error('Error connecting to MySQL:', error);
        process.exit(1);
    } finally {
        await connection.end();
    }
}

async function main() {
    const sampleText = `我最近认识了几个新朋友，第一个是张总，他是阿里巴巴的工程师，生日是1990-01-01，公司是阿里巴巴，职位是工程师，电话是13800138000，微信是zhangsan。第二个是李总，他是腾讯的工程师，生日是1991-02-02，公司是腾讯，职位是工程师，电话是13800138001，微信是lisi。第三个是王总，他是字节跳动的工程师，生日是1992-03-03，公司是字节跳动，职位是工程师，电话是13800138002，微信是wangwu。`;

    const result = await extractAndInsert(sampleText);
    console.log(result);
}

main()

