class VoiceAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.port.onmessage = (event) => {
            // No longer processing incoming audio through worklet
        };
    }

    process(inputs, outputs, parameters) {
        // 1. Process Input Audio from Microphone
        const input = inputs[0];
        if (input && input.length > 0 && input[0].length > 0) {
            const channelData = input[0];
            
            // Convert Float32 (browser standard) to Int16 (telephony/Pipecat standard)
            const pcmData = new Int16Array(channelData.length);
            for (let i = 0; i < channelData.length; i++) {
                let s = Math.max(-1, Math.min(1, channelData[i]));
                pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            
            // Send Int16 array back to main thread to send over WebSockets
            this.port.postMessage({ type: 'audio', audio: pcmData });
        }

        // We no longer manually fill the output buffers here, as app.js handles playback natively.
        // We still need to return true to keep the worklet alive for microphone capture!
        return true;
    }
}

registerProcessor('voice-audio-processor', VoiceAudioProcessor);
