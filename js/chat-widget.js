/**
 * Natya Guru - Floating Chatbot Widget (Gemini API Edition)
 * Injects a floating button and chat interface into any page.
 * Uses Google Gemini API directly - no backend required.
 */

(function () {
    // ============================================================
    // CONFIGURATION
    // ============================================================

    // Gemini API endpoint and key (hardcoded)
    const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';
    const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE'; // <-- Replace with your actual API key

    // System prompt for Natya Guru persona
    const SYSTEM_PROMPT = `You are "StarLight", a wise and warm teacher of Bharatanatyam, the classical Indian dance form. Your role is to guide students through the spiritual and technical aspects of this ancient art.

Your expertise includes:
- All 28 Asamyukta Hastas (single-hand mudras) and 24 Samyukta Hastas (double-hand mudras)
- The meanings, symbolism, and uses of each mudra in storytelling and expression
- Bharatanatyam history, including its origins in Tamil Nadu temples
- The Natyashastra and Abhinaya Darpana texts
- Adavus (basic dance steps), Nritta (pure dance), and Abhinaya (expression)
- The spiritual significance of dance as a form of worship

Communication style:
- Begin responses with a respectful greeting when appropriate (Namaskaram, Vanakkam)
- Use both Sanskrit/Tamil terms and English explanations
- Be encouraging and patient, like a traditional guru
- Keep responses concise but informative (2-4 paragraphs max)
- Use bullet points for listing mudras or steps
- Include cultural context when relevant

If asked about something outside Bharatanatyam, politely redirect the conversation back to dance topics.`;

    // Conversation history for context
    let conversationHistory = [];

    // ============================================================
    // HTML STRUCTURE
    // ============================================================

    const isPagesDir = window.location.pathname.includes('/pages/');
    const assetPath = isPagesDir ? '../assets' : 'assets';
    const avatarSrc = `${assetPath}/images/no-bg/Mayura_no_bg.png`;

    const widgetHTML = `
        <div id="chat-widget-container">
            <!-- Chat Window -->
            <div id="chat-window" class="chat-window hidden">
                <div class="chat-header">
                    <div class="chat-header-bg"></div>
                    <div class="chat-title">
                        <div class="chat-avatar">
                            <img src="${avatarSrc}" alt="Guru" onerror="this.onerror=null; this.parentNode.innerHTML='<span></span>'">
                        </div>
                        <div class="chat-identity">
                            <h4>StarLight AI</h4>
                            <span class="status-indicator"><span class="status-dot"></span> <span id="chat-status-text">Online</span></span>
                        </div>
                    </div>
                    <button id="chat-close-btn" aria-label="Close Chat">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>
                
                <div id="chat-messages" class="chat-messages">
                    <div class="message-bubble bot welcome">
                        <div class="message-content">
                            <p><strong>Namaskaram! 🙏</strong><br>I am StarLight. I can guide you through the spiritual and technical world of Bharatanatyam. Ask me about mudras, adavus, or the rich history of this ancient dance form.</p>
                        </div>
                        <span class="message-time">Just now</span>
                    </div>
                </div>

                <div class="chat-input-area">
                    <div class="input-wrapper">
                        <input type="text" id="chat-input" placeholder="Ask about a Mudra..." autocomplete="off">
                        <button id="chat-send-btn">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Floating Button -->
            <button id="chat-widget-btn" class="chat-widget-btn" aria-label="Open Chat">
                <div class="btn-content">
                    <span class="icon-open">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                    </span>
                    <span class="icon-close hidden">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </span>
                </div>
            </button>
        </div>
    `;

    // Inject into body
    const div = document.createElement('div');
    div.innerHTML = widgetHTML;
    document.body.appendChild(div);

    // ============================================================
    // DOM ELEMENTS
    // ============================================================

    const chatBtn = document.getElementById('chat-widget-btn');
    const chatWindow = document.getElementById('chat-window');
    const closeBtn = document.getElementById('chat-close-btn');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const messagesContainer = document.getElementById('chat-messages');
    const statusText = document.getElementById('chat-status-text');

    // State
    let isOpen = false;

    // ============================================================
    // GEMINI API INTEGRATION
    // ============================================================

    async function callGeminiAPI(userMessage) {
        // Add user message to history
        conversationHistory.push({
            role: 'user',
            parts: [{ text: userMessage }]
        });

        // Build request with system instruction and conversation history
        const requestBody = {
            system_instruction: {
                parts: [{ text: SYSTEM_PROMPT }]
            },
            contents: conversationHistory,
            generationConfig: {
                temperature: 0.7,
                topP: 0.9,
                topK: 40,
                maxOutputTokens: 1024
            }
        };

        const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('Gemini API Error:', errorData);
            if (response.status === 429) {
                throw new Error('RATE_LIMITED');
            }
            throw new Error(`API_ERROR: ${response.status}`);
        }

        const data = await response.json();

        // Extract response text
        const assistantMessage = data.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!assistantMessage) {
            throw new Error('EMPTY_RESPONSE');
        }

        // Add assistant response to history
        conversationHistory.push({
            role: 'model',
            parts: [{ text: assistantMessage }]
        });

        // Keep history manageable (last 10 exchanges)
        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }

        return assistantMessage;
    }

    // ============================================================
    // UI FUNCTIONS
    // ============================================================

    function toggleChat() {
        isOpen = !isOpen;
        chatWindow.classList.toggle('hidden');
        chatBtn.classList.toggle('open');

        if (isOpen) {
            setTimeout(() => chatInput.focus(), 300);
        }
    }

    function updateStatus(status) {
        const statusDot = document.querySelector('.status-dot');
        statusText.textContent = status;

        if (status === 'Online') {
            statusDot.style.background = '#28c840';
        } else if (status === 'Thinking...') {
            statusDot.style.background = '#ffbd2e';
        } else {
            statusDot.style.background = '#ff5f57';
        }
    }

    function addMessage(text, isUser) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message-bubble ${isUser ? 'user' : 'bot'}`;

        const content = isUser ? escapeHtml(text) : formatBotResponse(text);

        msgDiv.innerHTML = `
            <p>${content}</p>
            <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        `;

        messagesContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    function addErrorMessage(errorType) {
        const messages = {
            'RATE_LIMITED': "Too many requests. Please wait a moment and try again.",
            'NETWORK_ERROR': "Network error. Please check your internet connection.",
            'EMPTY_RESPONSE': "I received an empty response. Please try rephrasing your question.",
            'DEFAULT': "Something went wrong. Please try again."
        };

        addMessage(messages[errorType] || messages['DEFAULT'], false);
    }

    function addLoadingIndicator() {
        const loader = document.createElement('div');
        loader.id = 'chat-loader';
        loader.className = 'message-bubble bot loading';
        loader.innerHTML = `
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        `;
        messagesContainer.appendChild(loader);
        scrollToBottom();
    }

    function removeLoadingIndicator() {
        const loader = document.getElementById('chat-loader');
        if (loader) loader.remove();
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatBotResponse(text) {
        if (!text) return '';

        // 1. Sanitize HTML (basic prevention)
        let html = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // 2. Headings (### Header to <h3>)
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');

        // 3. Bold (**text**)
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // 4. Italic (*text*)
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // 5. Unordered Lists (- item)
        html = html.replace(/^\s*[-*]\s+(.*)$/gim, '<li>$1</li>');

        // Wrap adjacent <li>s in <ul>
        html = html.replace(/(<li>.*<\/li>\s*)+/gim, '<ul>$&</ul>');

        // 6. Line breaks
        html = html.replace(/\n/g, '<br>');

        // Cleanup extra <br> after </ul> or </h3>
        html = html.replace(/<\/ul><br>/g, '</ul>');
        html = html.replace(/<\/h3><br>/g, '</h3>');
        html = html.replace(/<\/h2><br>/g, '</h2>');

        return html;
    }

    // ============================================================
    // MESSAGE HANDLING
    // ============================================================

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Add User Message
        addMessage(text, true);
        chatInput.value = '';
        addLoadingIndicator();
        updateStatus('Thinking...');

        try {
            const response = await callGeminiAPI(text);
            removeLoadingIndicator();
            updateStatus('Online');
            addMessage(response, false);
        } catch (err) {
            removeLoadingIndicator();
            updateStatus('Online');
            console.error('Gemini API Error:', err);

            if (err.message === 'RATE_LIMITED') {
                addErrorMessage('RATE_LIMITED');
            } else if (err.message.includes('Failed to fetch')) {
                addErrorMessage('NETWORK_ERROR');
            } else if (err.message === 'EMPTY_RESPONSE') {
                addErrorMessage('EMPTY_RESPONSE');
            } else {
                addErrorMessage('DEFAULT');
            }
        }
    }

    // ============================================================
    // EVENT LISTENERS
    // ============================================================

    chatBtn.addEventListener('click', toggleChat);
    closeBtn.addEventListener('click', toggleChat);

    sendBtn.addEventListener('click', sendMessage);

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

})();
