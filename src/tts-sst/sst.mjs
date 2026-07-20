import "dotenv/config";
import tencentcloud from "tencentcloud-sdk-nodejs";
import fs from "node:fs";
import path from "node:path";

const SECRET_ID = process.env.TENCENT_CLOUD_SECRET_ID;
const SECRET_KEY = process.env.TENCENT_CLOUD_SECRET_KEY;

const AsrClient = tencentcloud.asr.v20190614.Client;
const AUDIO_FILE = path.join(process.cwd(), "src", "tts-sst", "tts.mp3");

const client = new AsrClient({
    credential: {
        secretId: SECRET_ID,
        secretKey: SECRET_KEY,
    },
    region: "ap-shanghai",
    profile: {
        httpProfile: {
            reqMethod: "POST",
            reqTimeout: 30,
        },
    },
});

async function run() {
    const audioBase64 = fs.readFileSync(AUDIO_FILE).toString("base64");

    const params = {
        EngSerViceType: "16k_zh",
        SourceType: 1,
        Data: audioBase64,
        DataLen: Buffer.byteLength(audioBase64),
        VoiceFormat: "mp3",
    };

    try {
        const data = await client.SentenceRecognition(params);
        console.log("识别结果：", data.Result);
    } catch (err) {
        console.error("识别失败：", err);
    }
}

run();