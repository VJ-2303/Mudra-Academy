/**
 * Ghost Mode - Local ML-Powered Practice
 * Uses local MediaPipe + ONNX models for mudra detection (no backend required)
 */

document.addEventListener('DOMContentLoaded', async () => {
    // ============================================================
    // DOM Elements
    // ============================================================
    const video = document.getElementById('webcam');
    const ghostImg = document.getElementById('ghost-overlay-img');
    const mudraNameEl = document.getElementById('current-mudra-name');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const voiceIndicator = document.getElementById('voice-indicator');
    const voiceStatusText = document.getElementById('voice-status-text');
    const canvas = document.getElementById('detection-canvas');
    const matchFeedback = document.getElementById('match-feedback');
    const debugStatus = document.getElementById('debug-status');

    // ============================================================
    // Configuration
    // ============================================================
    const DETECTION_INTERVAL = 150; // ms between detections
    const MATCH_THRESHOLD = 4; // Consecutive matches needed to advance
    let detectionTimer = null;
    let matchCounter = 0;

    // ============================================================
    // Mudra List (matches assets)
    // ============================================================
    const mudras = [
        'Pataka', 'Tripataka', 'Ardhapataka', 'Kartarimukha', 'Mayura',
        'Ardhachandra', 'Arala', 'Sukatunda', 'Musti', 'Sikharam',
        'Kapittha', 'Katakamukha', 'Suchi', 'Chandrakala', 'Padmakosa',
        'Sarpashirsa', 'Simhamukha', 'Kangula', 'Alapadma',
        'Chatura', 'Bhamara', 'Hamsasya', 'Hamsapaksa', 'Sandamsha',
        'Mukula', 'Tamracuda', 'Trisula'
    ];

    let currentIndex = 0;
    let ghostEnabled = true;
    let isVoiceActive = true;
    let isModelReady = false;

    // MediaPipe Hands instance
    let hands = null;
    let camera = null;

    // FSM instance for stable detection
    let mudraFSM = null;

    // ============================================================
    // Asset Path Helper
    // ============================================================
    const getAssetPath = (mudraName) => {
        let filename = mudraName;
        if (mudraName === 'Chatura') filename = 'Chautra';
        return `../assets/images/no-bg/${filename}_no_bg.png`;
    };

    // ============================================================
    // Initialize Application
    // ============================================================
    async function init() {
        try {
            debugStatus.textContent = "Loading ML models...";

            // Initialize ML models (RF for single-hand detection)
            await window.MudraInference.initializeRFOnly();

            // Create FSM instance
            mudraFSM = new window.MudraDetection.MudraFSM();

            isModelReady = true;
            debugStatus.textContent = "Models loaded! Starting camera...";

            // Initialize MediaPipe Hands
            await initMediaPipe();

            // Setup controls
            setupKeyboardControls();
            setupVoiceRecognition();

            // Initial display
            updateDisplay();

            setTimeout(() => {
                flashFeedback("Say 'Next', 'Back', or 'Ghost Off'");
            }, 1000);

        } catch (err) {
            console.error("Initialization error:", err);
            debugStatus.textContent = "Error: " + err.message;
            alert("Failed to initialize: " + err.message);
        }
    }

    // ============================================================
    // MediaPipe Hands Setup
    // ============================================================
    async function initMediaPipe() {
        // Load MediaPipe Hands
        hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });

        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });

        hands.onResults(onHandResults);

        // Setup camera
        camera = new Camera(video, {
            onFrame: async () => {
                if (hands && isModelReady) {
                    await hands.send({ image: video });
                }
            },
            width: 640,
            height: 480
        });

        await camera.start();
        debugStatus.textContent = "Ready! Show your mudra";
    }

    // ============================================================
    // Hand Detection Results Handler
    // ============================================================
    let prevLandmarks = null;

    function onHandResults(results) {
        if (!isModelReady) return;

        // Draw on canvas for visual feedback
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            const handedness = results.multiHandedness?.[0]?.label || "Right";

            // Draw hand landmarks
            drawHandLandmarks(ctx, landmarks, canvas.width, canvas.height);

            // Detect mudra using hybrid detection
            const detection = window.MudraDetection.detectMudraHybrid(
                landmarks,
                handedness,
                prevLandmarks
            );

            // Update FSM for stable detection
            const fsmResult = mudraFSM.update(
                true,
                detection.mudraName,
                detection.confidence,
                detection.method
            );

            prevLandmarks = landmarks;

            // Update debug display
            if (fsmResult.displayName && fsmResult.displayName !== "Detecting...") {
                const confPercent = Math.round(fsmResult.displayConf * 100);
                debugStatus.textContent = `${fsmResult.displayName} (${confPercent}%) [${fsmResult.displayMethod}]`;

                // Check for match with target
                checkMatch(fsmResult.displayName, fsmResult.displayConf);
            } else {
                debugStatus.textContent = "Detecting...";
            }
        } else {
            // No hand detected
            mudraFSM.update(false, null, 0, null);
            debugStatus.textContent = "Show your hand";
            prevLandmarks = null;

            // Reset match counter if no hand
            if (matchCounter > 0) {
                matchCounter = 0;
                video.style.boxShadow = "none";
            }
        }
    }

    // ============================================================
    // Draw Hand Landmarks
    // ============================================================
    function drawHandLandmarks(ctx, landmarks, width, height) {
        // Draw connections
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],     // Index
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring
            [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
            [5, 9], [9, 13], [13, 17]            // Palm
        ];

        ctx.strokeStyle = 'rgba(212, 168, 75, 0.8)';
        ctx.lineWidth = 2;

        for (const [i, j] of connections) {
            const x1 = landmarks[i].x * width;
            const y1 = landmarks[i].y * height;
            const x2 = landmarks[j].x * width;
            const y2 = landmarks[j].y * height;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }

        // Draw points
        for (let i = 0; i < landmarks.length; i++) {
            const x = landmarks[i].x * width;
            const y = landmarks[i].y * height;

            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = i === 0 ? '#8B2942' : '#D4A84B';
            ctx.fill();
        }
    }

    // ============================================================
    // Match Checking
    // ============================================================
    function checkMatch(detectedMudra, confidence) {
        if (!detectedMudra || confidence < 0.5) {
            if (matchCounter > 0) {
                matchCounter = 0;
                video.style.boxShadow = "none";
            }
            return;
        }

        const targetMudra = mudras[currentIndex];
        const d = normalizeMudraName(detectedMudra);
        const t = normalizeMudraName(targetMudra);

        const isMatch = (d === t) || d.includes(t) || t.includes(d);

        if (isMatch) {
            matchCounter++;

            // Visual feedback
            if (matchCounter === 1) {
                video.style.boxShadow = "0 0 15px var(--color-warning)";
            } else if (matchCounter >= 2) {
                video.style.boxShadow = "0 0 25px var(--color-secondary)";
            }

            // Update progress
            const progress = Math.min(100, Math.round((matchCounter / MATCH_THRESHOLD) * 100));
            debugStatus.textContent = `${detectedMudra} ✓ (${progress}% matched)`;

            if (matchCounter >= MATCH_THRESHOLD) {
                triggerSuccess();
            }
        } else {
            if (matchCounter > 0) {
                matchCounter = 0;
                video.style.boxShadow = "none";
            }
        }
    }

    function normalizeMudraName(name) {
        if (!name) return "";

        return name.toLowerCase()
            .replace(" mudra", "")
            .replace(/\s+/g, "")
            .replace('musthi', 'musti')
            .replace('shikharam', 'sikharam')
            .replace('shukatundam', 'sukatunda')
            .replace('kartarimukham', 'kartarimukha')
            .replace('trishula', 'trisula')
            .replace('sarpashirsha', 'sarpashirsa')
            .replace('chautra', 'chatura')
            .trim();
    }

    function triggerSuccess() {
        matchCounter = 0;
        video.style.boxShadow = "0 0 40px #28c840";
        matchFeedback.style.opacity = 1;

        // Play success sound (optional)
        try {
            const audio = new Audio('data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU');
        } catch (e) { }

        setTimeout(() => {
            matchFeedback.style.opacity = 0;
            video.style.boxShadow = "none";
            nextMudra();
        }, 600);
    }

    // ============================================================
    // Display & Navigation
    // ============================================================
    function updateDisplay() {
        const currentMudra = mudras[currentIndex];
        mudraNameEl.textContent = currentMudra;
        ghostImg.src = getAssetPath(currentMudra);
        ghostImg.style.opacity = ghostEnabled ? 0.35 : 0;
        speakMudra(currentMudra);
    }

    function nextMudra() {
        currentIndex = (currentIndex + 1) % mudras.length;
        matchCounter = 0;
        updateDisplay();
        flashFeedback('Next');
    }

    function prevMudra() {
        currentIndex = (currentIndex - 1 + mudras.length) % mudras.length;
        matchCounter = 0;
        updateDisplay();
        flashFeedback('Previous');
    }

    function toggleGhost(enable) {
        ghostEnabled = enable;
        ghostImg.style.opacity = enable ? 0.35 : 0;
        flashFeedback(enable ? 'Ghost On' : 'Ghost Off');
    }

    function speakMudra(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.9;
            utterance.pitch = 1.0;
            utterance.lang = 'hi-IN';

            const voices = window.speechSynthesis.getVoices();
            const hindiVoice = voices.find(v =>
                v.lang.includes('hi') || v.name.includes('India')
            );
            if (hindiVoice) utterance.voice = hindiVoice;

            window.speechSynthesis.speak(utterance);
        }
    }

    function flashFeedback(text) {
        if (!voiceStatusText.textContent.includes("Error")) {
            voiceStatusText.textContent = text;
            voiceStatusText.style.color = 'var(--color-primary)';

            setTimeout(() => {
                if (isVoiceActive) voiceStatusText.textContent = "Listening...";
                voiceStatusText.style.color = '';
            }, 2000);
        }
    }

    // ============================================================
    // Keyboard Controls
    // ============================================================
    function setupKeyboardControls() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === ' ' || e.key.toLowerCase() === 'n') {
                nextMudra();
            } else if (e.key === 'ArrowLeft' || e.key === 'Backspace' || e.key.toLowerCase() === 'b') {
                prevMudra();
            } else if (e.key.toLowerCase() === 'g') {
                toggleGhost(!ghostEnabled);
            }
        });
    }

    // ============================================================
    // Voice Recognition
    // ============================================================
    function setupVoiceRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            voiceStatusText.textContent = "Voice not supported (Use Chrome)";
            const badge = document.getElementById('voice-indicator-badge');
            if (badge) badge.classList.add('inactive');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            voiceStatusText.textContent = "Listening...";
            const badge = document.getElementById('voice-indicator-badge');
            if (badge) badge.classList.remove('inactive');
            isVoiceActive = true;
        };

        recognition.onerror = (event) => {
            if (event.error === 'no-speech') return;
            console.error("Speech error:", event.error);
            voiceStatusText.textContent = "Mic Off (Click to On)";
            const badge = document.getElementById('voice-indicator-badge');
            if (badge) badge.classList.add('inactive');
            isVoiceActive = false;
        };

        recognition.onend = () => {
            if (isVoiceActive) {
                try { recognition.start(); } catch (e) { }
            }
        };

        recognition.onresult = (event) => {
            const command = event.results[event.results.length - 1][0].transcript.trim().toLowerCase();
            console.log("Voice:", command);

            if (command.includes('next') || command.includes('forward')) {
                nextMudra();
            } else if (command.includes('back') || command.includes('previous')) {
                prevMudra();
            } else if (command.includes('ghost off') || command.includes('hide')) {
                toggleGhost(false);
            } else if (command.includes('ghost on') || command.includes('show')) {
                toggleGhost(true);
            }
        };

        try { recognition.start(); } catch (e) { console.error(e); }

        const badge = document.getElementById('voice-indicator-badge');
        if (badge) {
            badge.addEventListener('click', () => {
                if (isVoiceActive) {
                    isVoiceActive = false;
                    recognition.stop();
                } else {
                    isVoiceActive = true;
                    recognition.start();
                }
            });
        }
    }

    // ============================================================
    // Event Listeners
    // ============================================================
    nextBtn.addEventListener('click', nextMudra);
    prevBtn.addEventListener('click', prevMudra);

    // ============================================================
    // Start Application
    // ============================================================
    init();
});
