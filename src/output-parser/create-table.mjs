import 'dotenv/config';
import mysql from 'mysql2/promise';

async function main() {
    const connectionConfig = {
        host: process.env.MYSQL_HOST,
        port: Number(process.env.MYSQL_PORT ?? 3306),
        user: process.env.MYSQL_USER,
        password: process.env.MYSQL_PASSWORD,
        database: process.env.MYSQL_DATABASE,
        multipleStatements: true,
    }
    const connection = await mysql.createConnection(connectionConfig);

    try {
        await connection.query(`CREATE DATABASE IF NOT EXISTS hello CHARSET utf8mb4 COLLATE utf8mb4_general_ci`);
        await connection.query(`USE hello`);

        await connection.query(`
            CREATE TABLE IF NOT EXISTS friends (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                gender ENUM('male', 'female') NOT NULL,
                birthday DATE NOT NULL,
                company VARCHAR(100) NOT NULL,
                title VARCHAR(100) NOT NULL,
                phone VARCHAR(20) NOT NULL,
                wechat VARCHAR(100) NOT NULL
            ) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4
        `)

        const insertSql = `
           INSERT INTO friends (
               name, gender, birthday, company, title, phone, wechat
           ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `

        const values = [
            '张三',
            'male',
            '1990-01-01',
            '阿里巴巴',
            '工程师',
            '13800138000',
            'zhangsan',
        ]

        const [result] = await connection.query(insertSql, values);
        console.log(`Inserted ${result.affectedRows} rows`);
    } catch (error) {
        console.error('Error connecting to MySQL:', error);
        process.exit(1);
    } finally {
        await connection.end();
    }
}

main();