import Redis from 'ioredis';

const redis = new Redis({
    host: 'localhost',
    port: 6379,
    db: 0,
})

redis.on('connect', () => {
    console.log('Connected to Redis');
})

redis.on('error', (err) => {
    console.error('Redis error:', err);
})

redis.on('reconnecting', () => {
    console.log('Reconnecting to Redis');
})

async function main() {
    await redis.set('name', '张三')
    await redis.set('age', 20)
    console.log(await redis.get('name'))

    await redis.hset('user:1001', { name: '张三', age: 20 })
    console.log(await redis.hgetall('user:1001'))

    await redis.lpush('user:1001:friends', '李四')
    await redis.lpush('user:1001:friends', '王五')
    // 从左到右获取列表中的所有元素，0表示第一个元素，-1表示最后一个元素
    console.log(await redis.lrange('user:1001:friends', 0, -1))

    // 添加多个元素到集合中
    await redis.sadd('user:1001:tags', '程序员')
    await redis.sadd('user:1001:tags', '工程师')
    // 获取集合中的所有元素
    console.log(await redis.smembers('user:1001:tags'))

    // 添加多个元素到有序集合中
    await redis.zadd('user:1001:scores', 100, '张三', 90, '李四')
    console.log(await redis.zrange('user:1001:scores', 0, -1))

    const lockKey = 'my-lock'
    // 设置锁的过期时间10秒，如果锁不存在，则设置锁并返回OK
    const lockResult = await redis.set(lockKey, '1', 'EX', 10, 'NX')
    console.log(lockResult === 'OK' ? '获取锁成功' : '获取锁失败')
}

main()