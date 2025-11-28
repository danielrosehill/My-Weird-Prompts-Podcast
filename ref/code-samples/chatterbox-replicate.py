import { writeFile } from "fs/promises";
import Replicate from "replicate";
const replicate = new Replicate();

const input = {
    prompt: "We're excited to introduce Chatterbox, our first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.\n\nWhether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support emotion exaggeration control, a powerful feature that makes your voices stand out. Try it now on our Hugging Face Gradio app.\n\nIf you like the model but need to scale or finetune it for higher accuracy, check out our competitively priced TTS service (link). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media."
};

const output = await replicate.run("resemble-ai/chatterbox", { input });

// To access the file URL:
console.log(output.url());
//=> "https://replicate.delivery/.../output.wav"

// To write the file to disk:
await writeFile("output.wav", output);
//=> output.wav written to disk