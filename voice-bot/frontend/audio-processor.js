class VoiceAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        this.playbackQueue = [];
        this.playbackQueueSamples = 0;

        this.playbackStarted = false;
        this.playbackReadOffset = 0;

        this.playbackSampleRate = 16000;

        this.startBufferSamples = Math.floor(
            this.playbackSampleRate * 0.1
        );

        this.port.onmessage = (event) => {
            const data = event.data || {};

            // --------------------------------------------------------
            // New TTS audio
            // --------------------------------------------------------
            if (data.type === "playback") {
                this.playbackSampleRate =
                    Number(data.sampleRate) || 16000;

                const startBufferMs =
                    Number(data.startBufferMs) || 100;

                this.startBufferSamples = Math.max(
                    0,
                    Math.floor(
                        (this.playbackSampleRate * startBufferMs) /
                            1000
                    )
                );

                const pcm = data.audio;

                if (
                    !(pcm instanceof Int16Array) ||
                    pcm.length === 0
                ) {
                    return;
                }

                const samples = new Float32Array(
                    pcm.length
                );

                for (let i = 0; i < pcm.length; i++) {
                    samples[i] = pcm[i] / 32768.0;
                }

                this.playbackQueue.push(samples);

                this.playbackQueueSamples +=
                    samples.length;

                if (
                    !this.playbackStarted &&
                    this.playbackQueueSamples >=
                        this.startBufferSamples
                ) {
                    this.playbackStarted = true;
                }
            }

            // --------------------------------------------------------
            // HARD BARge-in reset
            // --------------------------------------------------------
            if (data.type === "clearPlayback") {
                this.playbackQueue.length = 0;
                this.playbackQueueSamples = 0;

                this.playbackStarted = false;
                this.playbackReadOffset = 0;
            }
        };
    }

    _writePlayback(outputChannel) {
        if (!this.playbackStarted) {
            outputChannel.fill(0);
            return;
        }

        let outputIndex = 0;

        while (
            outputIndex < outputChannel.length
        ) {
            if (
                this.playbackQueue.length === 0
            ) {
                outputChannel.fill(
                    0,
                    outputIndex
                );
                return;
            }

            const current =
                this.playbackQueue[0];

            const available =
                current.length -
                this.playbackReadOffset;

            const needed =
                outputChannel.length -
                outputIndex;

            const count = Math.min(
                available,
                needed
            );

            outputChannel.set(
                current.subarray(
                    this.playbackReadOffset,
                    this.playbackReadOffset +
                        count
                ),
                outputIndex
            );

            this.playbackReadOffset += count;
            outputIndex += count;

            this.playbackQueueSamples -=
                count;

            if (
                this.playbackReadOffset >=
                current.length
            ) {
                this.playbackQueue.shift();
                this.playbackReadOffset = 0;
            }
        }

        // Playback has caught up with the queue.
        // Wait for the next meaningful buffer.
        if (
            this.playbackQueueSamples === 0
        ) {
            this.playbackStarted = false;
        }
    }

    process(inputs, outputs) {
        // ------------------------------------------------------------
        // MICROPHONE INPUT
        // ------------------------------------------------------------
        const input = inputs[0];

        if (
            input &&
            input.length > 0 &&
            input[0] &&
            input[0].length > 0
        ) {
            const channelData = input[0];

            const pcmData = new Int16Array(
                channelData.length
            );

            for (
                let i = 0;
                i < channelData.length;
                i++
            ) {
                const sample = Math.max(
                    -1,
                    Math.min(
                        1,
                        channelData[i]
                    )
                );

                pcmData[i] =
                    sample < 0
                        ? sample * 0x8000
                        : sample * 0x7fff;
            }

            this.port.postMessage(
                {
                    type: "audio",
                    audio: pcmData,
                },
                [pcmData.buffer]
            );
        }

        // ------------------------------------------------------------
        // TTS PLAYBACK
        // ------------------------------------------------------------
        const output = outputs[0];

        if (
            output &&
            output.length > 0
        ) {
            const outputChannel =
                output[0];

            if (outputChannel) {
                this._writePlayback(
                    outputChannel
                );
            }
        }

        return true;
    }
}

registerProcessor(
    "voice-audio-processor",
    VoiceAudioProcessor
);