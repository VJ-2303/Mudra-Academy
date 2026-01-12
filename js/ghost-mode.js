document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const video = document.getElementById('webcam');
    const ghostImg = document.getElementById('ghost-overlay-img');
    const mudraNameEl = document.getElementById('current-mudra-name');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    // const opacitySlider = document.getElementById('opacity-slider'); // Removed
    const voiceIndicator = document.getElementById('voice-indicator');
    const voiceStatusText = document.getElementById('voice-status-text');
    const detectionCanvas = document.getElementById('detection-canvas');
    const matchFeedback = document.getElementById('match-feedback');
    const debugStatus = document.getElementById('debug-status');

    // API Configuration
    const API_URL = 'http://localhost:5000/detect';
    const DETECTION_INTERVAL = 200; // ms
    let detectionTimer = null;
    let matchCounter = 0;
    const MATCH_THRESHOLD = 3; // Number of consecutive matches to advance (approx 600ms-1s)

    // State
    // List of mudras based on available assets
    const mudras = [
        'Pataka', 'Tripataka', 'Ardhapataka', 'Kartarimukha', 'Mayura',
        'Ardhachandra', 'Arala', 'Sukatunda', 'Musti', 'Sikharam',
        'Kapittha', 'Katakamukha', 'Suchi', 'Chandrakala', 'Padmakosa',
        'Sarpashirsa', 'Simhamukha', 'Kangula', 'Alapadma', // Removed Mrugashirsha-Hasta
        'Chatura', 'Bhamara', 'Hamsasya', 'Hamsapaksa', 'Sandamsha',
        'Mukula', 'Tamracuda', 'Trisula'
    ]; // Note: "Chatura" filename is "Chautra.jpg" in assets, need to handle quirks

    let currentIndex = 0;
    let ghostEnabled = true; // State to track user preference
    let isVoiceActive = true; // State to track voice intention

    // Asset mapping to handle filename discrepancies
    const getAssetPath = (mudraName) => {
        let filename = mudraName;
        if (mudraName === 'Chatura') filename = 'Chautra';
        // Use no-bg images
        return `../assets/images/no-bg/${filename}_no_bg.png`;
    };

    // Initialize
    async function init() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;

            updateDisplay();
            updateDisplay();
            setupVoiceRecognition();
            setupKeyboardControls(); // Add keyboard support
            startDetectionLoop();    // Start AI detection

            // Initial feedback
            setTimeout(() => {
                flashFeedback("Say 'Next', 'Back', or 'Ghost Off'");
            }, 1000);

        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam. Please allow camera permissions.");
        }
    }

    function speakMudra(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.9; // Slightly slower for clarity
            utterance.pitch = 1.0;
            utterance.lang = 'hi-IN';

            // Try to find a specific Hindi/Indian voice
            const voices = window.speechSynthesis.getVoices();
            const hindiVoice = voices.find(v => v.lang.includes('hi') || v.name.includes('India') || v.name.includes('Hindi'));

            if (hindiVoice) {
                utterance.voice = hindiVoice;
            }

            window.speechSynthesis.speak(utterance);
        }
    }

    function updateDisplay() {
        const currentMudra = mudras[currentIndex];
        mudraNameEl.textContent = currentMudra;
        ghostImg.src = getAssetPath(currentMudra);
        // Respect the user's toggle choice
        ghostImg.style.opacity = ghostEnabled ? 0.3 : 0;

        // Speak the name
        speakMudra(currentMudra);
    }

    function nextMudra() {
        currentIndex = (currentIndex + 1) % mudras.length;
        updateDisplay();
        flashFeedback('Next');
    }

    function prevMudra() {
        currentIndex = (currentIndex - 1 + mudras.length) % mudras.length;
        updateDisplay();
        flashFeedback('Previous');
    }

    function toggleGhost(enable) {
        ghostEnabled = enable;
        updateDisplay();
        flashFeedback(enable ? 'Ghost On' : 'Ghost Off');
    }

    function flashFeedback(text) {
        const originalText = voiceStatusText.textContent;
        // Don't overwrite if it's an error message
        if (!originalText.includes("Error")) {
            voiceStatusText.textContent = `${text}`;
            voiceStatusText.style.color = 'var(--color-primary)';

            setTimeout(() => {
                // Check if still normal state before reverting
                if (isVoiceActive) voiceStatusText.textContent = "Listening...";
                voiceStatusText.style.color = '';
            }, 2000);
        }
    }

    // Event Listeners
    nextBtn.addEventListener('click', nextMudra);
    prevBtn.addEventListener('click', prevMudra);


    // Keyboard Controls
    function setupKeyboardControls() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === ' ' || e.key.toLowerCase() === 'n') {
                nextMudra();
            } else if (e.key === 'ArrowLeft' || e.key === 'Backspace' || e.key.toLowerCase() === 'b') {
                prevMudra();
            }
        });
    }


    // AI Detection Logic
    function startDetectionLoop() {
        if (detectionTimer) clearInterval(detectionTimer);

        // Set canvas size to match video source size once available
        video.addEventListener('loadedmetadata', () => {
            detectionCanvas.width = 640;
            detectionCanvas.height = 480;
        });

        detectionTimer = setInterval(detectMudra, DETECTION_INTERVAL);
    }

    async function detectMudra() {
        if (!video || !detectionCanvas) return;

        // Draw current video frame to canvas
        const ctx = detectionCanvas.getContext('2d');
        // Ensure consistent size
        if (detectionCanvas.width !== 640) detectionCanvas.width = 640;
        if (detectionCanvas.height !== 480) detectionCanvas.height = 480;

        ctx.drawImage(video, 0, 0, detectionCanvas.width, detectionCanvas.height);

        try {
            const imageBase64 = detectionCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageBase64 })
            });

            const data = await response.json();

            if (data.success) {
                // Update debug display
                const mudra = data.mudra || "None";
                const conf = data.confidence ? Math.round(data.confidence * 100) : 0;
                debugStatus.textContent = `API: ${mudra} (${conf}%)`;

                checkMatch(mudra);
            }
        } catch (err) {
            console.error("Detection error:", err);
            debugStatus.textContent = "API Error: " + err.message;
        }
    }

    // Helper: Normalize backend names to match frontend expectations
    function normalizeMudraName(backendName) {
        if (!backendName) return "";

        // 1. Basic cleanup: Lowercase, remove ' mudra', remove internal spaces
        let name = backendName.toLowerCase()
            .replace(" mudra", "")
            .replace(/\s+/g, "") // Remove all spaces (e.g. "shuka tundam" -> "shukatundam")
            .trim();

        // 2. Comprehensive Mapping (Keys must be lowercased & spaceless)
        const mapping = {
            'musthi': 'musti',
            'shikharam': 'sikharam',
            'shukatundam': 'sukatunda',
            'kartarimukham': 'kartarimukha',
            'trishula': 'trisula',
            'sarpashirsha': 'sarpashirsa',
            'mrigasheersha': 'mrugashirsha',
            'chautra': 'chatura',
            // Ensure identity mappings work if backend sends clean names
            'simhamukha': 'simhamukha'
        };

        return mapping[name] || name;
    }

    function checkMatch(detectedMudra) {
        const targetMudra = mudras[currentIndex];

        // Normalize strings for comparison
        const d = normalizeMudraName(detectedMudra);
        const t = targetMudra.toLowerCase().trim();

        console.log(`Checking match: API='${detectedMudra}' -> '${d}' vs Target='${t}' (Count: ${matchCounter})`);

        // Comparison (Exact or substring)
        const isMatch = (d === t) || (d.includes(t)) || (t.includes(d));

        if (isMatch) {
            matchCounter++;

            // Visual feedback building up (Yellow -> Orange)
            if (matchCounter === 1) {
                video.style.boxShadow = "0 0 10px var(--color-warning)";
            } else if (matchCounter >= 2) {
                video.style.boxShadow = "0 0 20px var(--color-secondary)";
            }

            // Update debug status with counter
            if (debugStatus) {
                const perc = Math.min(100, Math.round((matchCounter / MATCH_THRESHOLD) * 100));
                debugStatus.textContent = `Target: ${targetMudra} | Match: ${perc}%`;
            }

            if (matchCounter >= MATCH_THRESHOLD) {
                console.log("Match threshold reached! Triggering success.");
                triggerSuccess();
            }
        } else {
            // Reset if sequence broken
            if (matchCounter > 0) console.log("Match broken, resetting counter.");
            matchCounter = 0;
            video.style.boxShadow = "none";
        }
    }

    function triggerSuccess() {
        matchCounter = 0; // Reset
        video.style.boxShadow = "0 0 30px #28c840"; // Bright green
        matchFeedback.style.opacity = 1; // Show "Match!" badge

        // Optional: Sound effect could go here

        // Say "Correct" or just the next mudra name?
        // Let's just move to next, the next name will be spoken.

        setTimeout(() => {
            matchFeedback.style.opacity = 0;
            video.style.boxShadow = "none";
            nextMudra();
        }, 500); // Short delay to see the success state
    }

    // Voice Recognition Logic
    function setupVoiceRecognition() {
        // Feature check
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            voiceStatusText.textContent = "Voice not supported (Use Chrome)";
            voiceIndicator.style.backgroundColor = '#666';
            document.getElementById('voice-indicator-badge').classList.add('inactive');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = true; // Keep listening
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            voiceStatusText.textContent = "Listening...";
            document.getElementById('voice-indicator-badge').classList.remove('inactive');
            isVoiceActive = true;
        };

        recognition.onerror = (event) => {
            if (event.error === 'no-speech') return; // Ignore silence errors

            console.error("Speech recognition error", event.error);
            voiceStatusText.textContent = "Mic Off (Click to On)";
            document.getElementById('voice-indicator-badge').classList.add('inactive');
            isVoiceActive = false;
        };

        recognition.onend = () => {
            // Auto-restart if we think it should still be running
            if (isVoiceActive) {
                console.log("Voice service stopped, restarting...");
                try {
                    recognition.start();
                } catch (e) { /* ignore already started errors */ }
            } else {
                document.getElementById('voice-indicator-badge').classList.add('inactive');
                voiceStatusText.textContent = "Mic Off";
            }
        };

        recognition.onresult = (event) => {
            const lastResult = event.results[event.results.length - 1];
            const command = lastResult[0].transcript.trim().toLowerCase();

            console.log("Voice command received:", command);

            if (command.includes('next') || command.includes('forward') || command.includes('go')) {
                nextMudra();
            } else if (command.includes('back') || command.includes('previous') || command.includes('prev')) {
                prevMudra();
            } else if (command.includes('ghost off') || command.includes('hide ghost')) {
                toggleGhost(false);
            } else if (command.includes('ghost on') || command.includes('show ghost')) {
                toggleGhost(true);
            }
        };

        // Start listening
        try {
            recognition.start();
        } catch (e) {
            console.error(e);
        }

        // Toggle voice on click of status area
        document.getElementById('voice-indicator-badge').addEventListener('click', () => {
            if (isVoiceActive) {
                isVoiceActive = false;
                recognition.stop();
            } else {
                isVoiceActive = true;
                recognition.start();
            }
        });
    }

    // Run
    init();
});
