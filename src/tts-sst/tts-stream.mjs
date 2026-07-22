/**
 * 腾讯云流式文本语音合成 (TextToStreamAudioWSv2) Node.js 示例
 *
 * 基于文档: https://cloud.tencent.com/document/product/1073/108595
 */

import "dotenv/config";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import WebSocket from "ws";
import { v4 as uuidv4 } from "uuid";

const CONFIG = {
    AppId: Number(process.env.TENCENT_CLOUD_APP_ID),
    SecretId: process.env.TENCENT_CLOUD_SECRET_ID,
    SecretKey: process.env.TENCENT_CLOUD_SECRET_KEY,

    VoiceType: 101001,
    Volume: 0,
    Speed: 0,
    SampleRate: 16000,
    Codec: "mp3",
    EnableSubtitle: false,
};

const TEXT_TO_SYNTHESIZE =
    "欢迎使用腾讯云流式文本语音合成服务。这是一个基于 WebSocket 的实时语音合成接口，支持流式文本输入，适用于大语言模型的逐字输入场景。";

const OUTPUT_FILE = path.join(process.cwd(), "src", "tts-sst", "output.mp3");

/**
 * 生成 HMAC-SHA1 签名
 * 签名原文格式: GET + 域名 + ? + 按字典序排序的参数（参数值不做 URL 编码）
 */
const generateSignature = (params, secretKey) => {
    const host = "tts.cloud.tencent.com/stream_wsv2";

    const sortedKeys = Object.keys(params)
        .filter((k) => k !== "Signature")
        .sort();

    const paramStr = sortedKeys.map((key) => `${key}=${params[key]}`).join("&");
    const signStr = `GET${host}?${paramStr}`;

    console.log("[签名] 签名原文:", signStr);

    const signature = crypto.createHmac("sha1", secretKey).update(signStr).digest("base64");
    console.log("[签名] 签名值:", signature);

    return signature;
};

/**
 * 构建 WebSocket 连接 URL
 */
const buildWebSocketUrl = (config) => {
    const timestamp = Math.floor(Date.now() / 1000);
    const expired = timestamp + 86400;
    const sessionId = uuidv4();

    const params = {
        Action: "TextToStreamAudioWSv2",
        AppId: config.AppId,
        SecretId: config.SecretId,
        Timestamp: timestamp,
        Expired: expired,
        SessionId: sessionId,
        VoiceType: config.VoiceType,
        Volume: config.Volume,
        Speed: config.Speed,
        SampleRate: config.SampleRate,
        Codec: config.Codec,
        EnableSubtitle: config.EnableSubtitle,
    };

    if (config.EmotionCategory) {
        params.EmotionCategory = config.EmotionCategory;
    }
    if (config.EmotionIntensity !== undefined) {
        params.EmotionIntensity = config.EmotionIntensity;
    }
    if (config.SegmentRate !== undefined) {
        params.SegmentRate = config.SegmentRate;
    }

    const signature = generateSignature(params, config.SecretKey);
    // Signature 只做一次 URL 编码（不要先 encode 再二次 encode）
    params.Signature = signature;

    const queryString = Object.entries(params)
        .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
        .join("&");

    const url = `wss://tts.cloud.tencent.com/stream_wsv2?${queryString}`;

    console.log("\n[连接] WebSocket URL 已构建 (SessionId:", sessionId, ")");

    return { url, sessionId };
};

const sendAction = (ws, sessionId, action, data = "") => {
    ws.send(
        JSON.stringify({
            session_id: sessionId,
            message_id: uuidv4(),
            action,
            data,
        })
    );
};

const main = async () => {
    const { url, sessionId } = buildWebSocketUrl(CONFIG);
    const ws = new WebSocket(url);

    const audioChunks = [];
    let isReady = false;
    let isFinished = false;

    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            if (!isReady) {
                ws.close();
                reject(new Error("连接超时，未收到服务端 READY 事件"));
            }
        }, 15000);

        ws.on("open", () => {
            console.log("[连接] WebSocket 连接已建立");
        });

        // ws 默认把文本帧也以 Buffer 形式交付，必须用 isBinary 区分
        ws.on("message", (data, isBinary) => {
            if (isBinary) {
                audioChunks.push(data);
                console.log(`[音频] 收到音频数据块: ${data.length} 字节`);
                return;
            }

            try {
                const msg = JSON.parse(data.toString());
                console.log("[消息]", JSON.stringify(msg, null, 2));

                if (msg.code !== 0) {
                    console.error("[错误]", msg.message);
                    clearTimeout(timeout);
                    ws.close();
                    reject(new Error(`服务端错误 (${msg.code}): ${msg.message}`));
                    return;
                }

                if (msg.ready === 1) {
                    isReady = true;
                    clearTimeout(timeout);
                    console.log("[就绪] 服务端已就绪，开始发送合成文本...");

                    sendAction(ws, sessionId, "ACTION_SYNTHESIS", TEXT_TO_SYNTHESIZE);
                    console.log("[发送] ACTION_SYNTHESIS:", `${TEXT_TO_SYNTHESIZE.substring(0, 50)}...`);

                    sendAction(ws, sessionId, "ACTION_COMPLETE");
                    console.log("[发送] ACTION_COMPLETE");
                }

                if (msg.final === 1) {
                    isFinished = true;
                    console.log("[完成] 合成结束，关闭连接");

                    const audioBuffer = Buffer.concat(audioChunks);
                    fs.writeFileSync(OUTPUT_FILE, audioBuffer);
                    console.log(`[保存] 音频已保存到: ${OUTPUT_FILE} (${audioBuffer.length} 字节)`);

                    ws.close();
                }

                if (msg.result?.subtitles) {
                    console.log("[时间戳] 词数:", msg.result.subtitles.length);
                }

                if (msg.heartbeat === 1) {
                    console.log("[心跳] 收到心跳报文");
                }
            } catch (err) {
                console.error("[解析] 消息解析失败:", err.message);
            }
        });

        ws.on("close", (code) => {
            console.log(`[关闭] WebSocket 连接已关闭 (code: ${code})`);

            if (isFinished) {
                resolve(audioChunks);
            } else if (!isReady) {
                reject(new Error("连接在握手阶段关闭"));
            } else {
                reject(new Error(`连接在合成阶段异常关闭 (code: ${code})`));
            }
        });

        ws.on("error", (err) => {
            console.error("[错误] WebSocket 错误:", err.message);
            clearTimeout(timeout);
            reject(err);
        });
    });
};

try {
    const chunks = await main();
    const totalSize = chunks.reduce((sum, c) => sum + c.length, 0);
    console.log(`\n✅ 流式语音合成成功！总音频大小: ${totalSize} 字节`);
    console.log(`   音频格式: ${CONFIG.Codec.toUpperCase()}`);
    console.log(`   采样率: ${CONFIG.SampleRate} Hz`);
    console.log(`   输出文件: ${OUTPUT_FILE}`);

    console.log(`\n💡 播放: open ${OUTPUT_FILE}`);
} catch (err) {
    console.error("\n❌ 合成失败:", err.message);
    process.exit(1);
}
