let ws = null;
let audioContext = null;
let mediaStream = null;
let workletNode = null;
let isConnected = false;

// Audio scheduling trackers
let nextPlayTime = 0;
let activeSourceNodes = [];

const callBtn = document.getElementById('callButton');
const callBtnText = document.getElementById('callButtonText');
const statusText = document.getElementById('status');
const waveform = document.getElementById('waveform');

// Use current host to guess websocket URL. Testing locally uses localhost:8000
const WS_URL = "ws://localhost:8000/ws-web";

// Utility: convert Int16Array to Base64
function int16ArrayToBase64(int16Array) {
    const uint8Array = new Uint8Array(int16Array.buffer);
    let binary = '';
    for (let i = 0; i < uint8Array.byteLength; i++) {
        binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary);
}

// Utility: convert Base64 to Float32Array (for playing back)
function base64ToFloat32Array(base64) {
    const binary = atob(base64);
    const len = binary.length;
    const safeLen = len % 2 === 0 ? len : len - 1; // Ensure 16-bit alignment
    
    const bytes = new Uint8Array(safeLen);
    for (let i = 0; i < safeLen; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    
    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768.0;
    }
    return float32;
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
                autoGainControl: true
            }
        });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        
        // Ensure context isn't suspended (Safari/Chrome strict autoplay policy fix)
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        
        await audioContext.audioWorklet.addModule('audio-processor.js');
        const source = audioContext.createMediaStreamSource(mediaStream);
        workletNode = new AudioWorkletNode(audioContext, 'voice-audio-processor');
        
        source.connect(workletNode);
        workletNode.connect(audioContext.destination);

        statusText.innerText = `Connecting to voice agent...`;
        console.log(`AudioContext Initialized. State: ${audioContext.state}, hardware SR: ${audioContext.sampleRate}`);

        ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {
             isConnected = true;
             updateUIState(true);
             statusText.innerText = "Connected - Say Hello!";
             
             // Message from audio processor: audio captured from mic, send to WebSocket
             workletNode.port.onmessage = (event) => {
                 if (event.data.type === 'audio' && ws.readyState === WebSocket.OPEN) {
                     const base64Data = int16ArrayToBase64(event.data.audio);
                     ws.send(JSON.stringify({
                         event: "media",
                         media: { payload: base64Data }
                     }));
                 }
             };
        };

        ws.onmessage = (event) => {
             try {
                const data = JSON.parse(event.data);
                if (data.event === 'playAudio' && data.media && data.media.payload) {
                    const float32Data = base64ToFloat32Array(data.media.payload);
                    const sampleRate = data.media.sampleRate || 16000;
                    
                    // console.log(`Received playing buffer: len=${float32Data.length}, SR=${sampleRate}`);

                    if (audioContext.state === 'suspended') {
                        audioContext.resume();
                    }

                    // Schedule Native Audio Buffer
                    const audioBuffer = audioContext.createBuffer(1, float32Data.length, sampleRate);
                    audioBuffer.getChannelData(0).set(float32Data);
                    
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);
                    
                    const currentTime = audioContext.currentTime;
                    // Reset play time if the scheduled time is deeply in the past
                    if (nextPlayTime < currentTime) {
                        nextPlayTime = currentTime;
                    }
                    
                    source.start(nextPlayTime);
                    nextPlayTime += audioBuffer.duration;
                    
                    activeSourceNodes.push(source);
                    
                    // Cleanup finished sources from array
                    source.onended = () => {
                         const index = activeSourceNodes.indexOf(source);
                         if (index > -1) activeSourceNodes.splice(index, 1);
                    };

                    waveform.classList.add('active');
                    clearTimeout(window.waveformTimeout);
                    window.waveformTimeout = setTimeout(() => {
                         waveform.classList.remove('active');
                    }, Math.max(500, audioBuffer.duration * 1000)); 
                }
                if (data.event === 'clearAudio') {
                     // The LLM was interrupted by the user! Stop all playing audio instantly.
                     activeSourceNodes.forEach(s => {
                         try { s.stop(); } catch(e) {}
                     });
                     activeSourceNodes = [];
                     nextPlayTime = audioContext ? audioContext.currentTime : 0;
                }
             } catch(e) {
                console.error("Frame parse error:", e);
             }
        };

        ws.onclose = () => {
             disconnectCall();
             statusText.innerText = "Call ended.";
        };

        ws.onerror = (e) => {
             console.error("WebSocket Error:", e);
             statusText.innerText = "Connection error. Is the backend running?";
             disconnectCall();
        };

    } catch (err) {
        console.error("Initialization error:", err);
        statusText.innerText = "Microphone access denied or error occurred.";
    }
}

function disconnectCall() {
    isConnected = false;
    updateUIState(false);
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    if (workletNode) {
        workletNode.disconnect();
        workletNode = null;
    }
    
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    waveform.classList.remove('active');
    statusText.innerText = "Disconnected.";
}

function updateUIState(active) {
    if (active) {
        callBtn.classList.remove('start-call');
        callBtn.classList.add('end-call');
        callBtnText.innerText = "End Call";
    } else {
        callBtn.classList.remove('end-call');
        callBtn.classList.add('start-call');
        callBtnText.innerText = "Connect";
    }
}

// Attach listeners
callBtn.addEventListener('click', () => {
    if (!isConnected) {
        initCall();
    } else {
        disconnectCall();
    }
});

// Setup initial UI states
updateUIState(false);
