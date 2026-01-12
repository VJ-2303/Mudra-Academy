/**
 * Natya Guru - Floating Chatbot Widget
 * Injects a floating button and chat interface into any page.
 */

(function () {
    // Configuration
    const API_URL = 'http://localhost:5000/chat';

    // Inject CSS if not present (handled by update to styles.css, but we function autonomously)
    // We assume styles.css contains the necessary classes.

    // Determine base path for assets based on current location
    const isPagesDir = window.location.pathname.includes('/pages/');
    const assetPath = isPagesDir ? '../assets' : 'assets';
    const avatarSrc = `${assetPath}/images/no-bg/Mayura_no_bg.png`;

    // HTML Structure
    const widgetHTML = `
        <div id="chat-widget-container">
            <!-- Chat Window -->
            <div id="chat-window" class="chat-window hidden">
                <div class="chat-header">
                    <div class="chat-header-bg"></div>
                    <div class="chat-title">
                        <div class="chat-avatar">
                            <img src="${avatarSrc}" alt="Guru" onerror="this.onerror=null; this.parentNode.innerHTML='<span>🕉️</span>'">
                        </div>
                        <div class="chat-identity">
                            <h4>Natya Guru</h4>
                            <span class="status-indicator"><span class="status-dot"></span> Online</span>
                        </div>
                    </div>
                    <button id="chat-close-btn" aria-label="Close Chat">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>
                
                <div id="chat-messages" class="chat-messages">
                    <div class="message-bubble bot welcome">
                        <div class="message-content">
                            <p><strong>Namaskaram! 🙏</strong><br>I am Natya Guru. I can guide you through the spiritual and technical world of Bharatanatyam.</p>
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

    // Elements
    const chatBtn = document.getElementById('chat-widget-btn');
    const chatWindow = document.getElementById('chat-window');
    const closeBtn = document.getElementById('chat-close-btn');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const messagesContainer = document.getElementById('chat-messages');

    // State
    let isOpen = false;

    // Functions
    function toggleChat() {
        isOpen = !isOpen;
        chatWindow.classList.toggle('hidden');
        chatBtn.classList.toggle('open');

        if (isOpen) {
            setTimeout(() => chatInput.focus(), 300);
        }
    }

    function addMessage(text, isUser) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message-bubble ${isUser ? 'user' : 'bot'}`;

        const content = isUser ? text : formatBotResponse(text);

        msgDiv.innerHTML = `
            <p>${content}</p>
            <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        `;

        messagesContainer.appendChild(msgDiv);
        scrollToBottom();
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
        // Convert lines starting with "- " or "* " to list items
        // We wrap the whole block in <ul> if strictly adjacent, but for simplicity in a chat bubble:
        // just treating them as bullet lines with <br> helps, or better:
        html = html.replace(/^\s*[-*]\s+(.*)$/gim, '<li>$1</li>');

        // Wrap adjacent <li>s in <ul> (simple regex approach)
        // This regex looks for a sequence of <li>...</li> and wraps them.
        // It's not perfect but works for simple chat outputs.
        html = html.replace(/(<li>.*<\/li>\s*)+/gim, '<ul>$&</ul>');

        // 6. Line breaks
        // Replace newlines with <br>, but NOT inside <ul> or after headings
        html = html.replace(/\n/g, '<br>');

        // Cleanup extra <br> after </ul> or </h3>
        html = html.replace(/<\/ul><br>/g, '</ul>');
        html = html.replace(/<\/h3><br>/g, '</h3>');
        html = html.replace(/<\/h2><br>/g, '</h2>');

        return html;
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Add User Message
        addMessage(text, true);
        chatInput.value = '';
        addLoadingIndicator();

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });

            const data = await response.json();
            removeLoadingIndicator();

            if (data.success) {
                addMessage(data.reply, false);
            } else {
                addMessage("I'm having trouble connecting to my wisdom source. Please check the API Server.", false);
                console.error(data.error);
            }
        } catch (err) {
            removeLoadingIndicator();
            addMessage("Network error. Is the server running?", false);
            console.error(err);
        }
    }

    // Event Listeners
    chatBtn.addEventListener('click', toggleChat);
    closeBtn.addEventListener('click', toggleChat);

    sendBtn.addEventListener('click', sendMessage);

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

})();
