let ws = null;
let audioContext = null;
let mediaStream = null;
let workletNode = null;
let isConnected = false;

const PLAYBACK_START_BUFFER_MS = 100;

const callBtn = document.getElementById("callButton");
const callBtnText = document.getElementById("callButtonText");
const statusText = document.getElementById("status");
const waveform = document.getElementById("waveform");

const BACKEND_HOST = "localhost:8000";
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${WS_PROTOCOL}//${BACKEND_HOST}/ws-web`;

function int16ArrayToBase64(int16Array) {
    const uint8Array = new Uint8Array(
        int16Array.buffer,
        int16Array.byteOffset,
        int16Array.byteLength
    );

    let binary = "";
    const chunkSize = 0x8000;

    for (let offset = 0; offset < uint8Array.length; offset += chunkSize) {
        const chunk = uint8Array.subarray(
            offset,
            Math.min(offset + chunkSize, uint8Array.length)
        );
        binary += String.fromCharCode(...chunk);
    }

    return btoa(binary);
}

function base64ToInt16Array(base64) {
    const binary = atob(base64);
    const byteLength = binary.length - (binary.length % 2);
    const bytes = new Uint8Array(byteLength);

    for (let i = 0; i < byteLength; i++) {
        bytes[i] = binary.charCodeAt(i);
    }

    return new Int16Array(bytes.buffer);
}

async function initCall() {
    statusText.innerText = "Requesting microphone permission...";

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        audioContext = new (
            window.AudioContext || window.webkitAudioContext
        )({
            sampleRate: 16000,
        });

        if (audioContext.state === "suspended") {
            await audioContext.resume();
        }

        await audioContext.audioWorklet.addModule("audio-processor.js");

        const source = audioContext.createMediaStreamSource(mediaStream);

        workletNode = new AudioWorkletNode(
            audioContext,
            "voice-audio-processor",
            {
                numberOfInputs: 1,
                numberOfOutputs: 1,
                outputChannelCount: [1],
            }
        );

        source.connect(workletNode);
        workletNode.connect(audioContext.destination);

        workletNode.port.onmessage = (event) => {
            if (
                event.data?.type === "audio" &&
                ws &&
                ws.readyState === WebSocket.OPEN
            ) {
                const base64Data = int16ArrayToBase64(event.data.audio);

                ws.send(
                    JSON.stringify({
                        event: "media",
                        media: {
                            payload: base64Data,
                            sampleRate: 16000,
                        },
                    })
                );
            }
        };

        statusText.innerText = "Connecting to voice agent...";

        console.log(
            `AudioContext initialized: state=${audioContext.state}, sampleRate=${audioContext.sampleRate}`
        );

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            isConnected = true;
            updateUIState(true);
            statusText.innerText = "Connected - Say Hello!";
        };

        ws.onmessage = async (event) => {
            try {
                const data = JSON.parse(event.data);

                if (
                    data.event === "playAudio" &&
                    data.media &&
                    data.media.payload
                ) {
                    if (!audioContext || audioContext.state === "closed") {
                        return;
                    }

                    if (audioContext.state === "suspended") {
                        await audioContext.resume();
                    }

                    const sampleRate = Number(
                        data.media.sampleRate || audioContext.sampleRate
                    );

                    if (sampleRate !== audioContext.sampleRate) {
                        console.warn(
                            `Received audio at ${sampleRate} Hz, browser context is ${audioContext.sampleRate} Hz`
                        );
                    }

                    const pcm = base64ToInt16Array(data.media.payload);

                    workletNode.port.postMessage(
                        {
                            type: "playback",
                            audio: pcm,
                            sampleRate,
                            startBufferMs: PLAYBACK_START_BUFFER_MS,
                        },
                        [pcm.buffer]
                    );

                    waveform.classList.add("active");
                    clearTimeout(window.waveformTimeout);
                    window.waveformTimeout = setTimeout(() => {
                        waveform.classList.remove("active");
                    }, PLAYBACK_START_BUFFER_MS + 250);
                }

                if (data.event === "clearAudio") {
                    if (workletNode) {
                        workletNode.port.postMessage({
                            type: "clearPlayback",
                        });
                    }

                    waveform.classList.remove("active");
                }
            } catch (error) {
                console.error("WebSocket audio frame error:", error);
            }
        };

        ws.onclose = () => {
            disconnectCall(false);
            statusText.innerText = "Call ended.";
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            statusText.innerText =
                "Connection error. Is the backend running?";
            disconnectCall(false);
        };
    } catch (error) {
        console.error("Initialization error:", error);
        statusText.innerText =
            "Microphone access denied or initialization failed.";
        disconnectCall(false);
    }
}

function disconnectCall(updateStatus = true) {
    isConnected = false;
    updateUIState(false);

    if (workletNode) {
        workletNode.port.postMessage({
            type: "clearPlayback",
        });
        workletNode.disconnect();
        workletNode = null;
    }

    if (ws) {
        try {
            ws.close();
        } catch (error) {
            console.debug("WebSocket close error:", error);
        }
        ws = null;
    }

    if (audioContext) {
        try {
            audioContext.close();
        } catch (error) {
            console.debug("AudioContext close error:", error);
        }
        audioContext = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop());
        mediaStream = null;
    }

    waveform.classList.remove("active");

    if (updateStatus) {
        statusText.innerText = "Disconnected.";
    }
}

function updateUIState(active) {
    if (active) {
        callBtn.classList.remove("start-call");
        callBtn.classList.add("end-call");
        callBtnText.innerText = "End Call";
    } else {
        callBtn.classList.remove("end-call");
        callBtn.classList.add("start-call");
        callBtnText.innerText = "Connect";
    }
}

callBtn.addEventListener("click", () => {
    if (!isConnected) {
        initCall();
    } else {
        disconnectCall();
    }
});

updateUIState(false);