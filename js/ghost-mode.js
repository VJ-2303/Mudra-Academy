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
            setupVoiceRecognition();
            setupKeyboardControls(); // Add keyboard support
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam. Please allow camera permissions.");
        }
    }

    function updateDisplay() {
        const currentMudra = mudras[currentIndex];
        mudraNameEl.textContent = currentMudra;
        ghostImg.src = getAssetPath(currentMudra);
        ghostImg.style.opacity = 0.3; // Fixed opacity
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

    function flashFeedback(text) {
        const originalText = voiceStatusText.textContent;
        voiceStatusText.textContent = `Heard: "${text}"`;
        voiceStatusText.style.color = 'var(--color-gold)';

        setTimeout(() => {
            voiceStatusText.textContent = originalText;
            voiceStatusText.style.color = '';
        }, 1500);
    }

    // Event Listeners
    nextBtn.addEventListener('click', nextMudra);
    prevBtn.addEventListener('click', prevMudra);

    // opacitySlider removed


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

    // Voice Recognition Logic
    function setupVoiceRecognition() {
        // Feature check
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            voiceStatusText.textContent = "Voice not supported (Use Chrome)";
            voiceIndicator.style.backgroundColor = '#666';

            // Show a helpful toast/message about keyboard controls as fallback
            const helper = document.querySelector('.voice-helper');
            if (helper) {
                const note = document.createElement('div');
                note.style.color = '#ff9999';
                note.style.marginTop = '10px';
                note.style.fontSize = '0.8rem';
                note.innerHTML = '⚠️ Firefox/Safari may not support Voice.<br>Use <b>Arrow Keys</b> keys to navigate.';
                helper.appendChild(note);
            }
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            voiceStatusText.textContent = "Listening...";
            document.getElementById('voice-indicator-badge').classList.remove('inactive');
            // voiceIndicator.classList.add('listening'); // Old logic
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error", event.error);
            voiceStatusText.textContent = "Error (Click to retry)";
            document.getElementById('voice-indicator-badge').classList.add('inactive');
        };

        recognition.onend = () => {
            // Auto restart
            // document.getElementById('voice-indicator-badge').classList.add('inactive');
            // voiceStatusText.textContent = "Mic off";
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
                // opacitySlider.value = 0;
                ghostImg.style.opacity = 0;
                flashFeedback('Ghost Off');
            } else if (command.includes('ghost on') || command.includes('show ghost')) {
                // opacitySlider.value = 0.4;
                ghostImg.style.opacity = 0.3;
                flashFeedback('Ghost On');
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
            try {
                recognition.start();
            } catch (e) {
                recognition.stop();
            }
        });
    }

    // Run
    init();
});
