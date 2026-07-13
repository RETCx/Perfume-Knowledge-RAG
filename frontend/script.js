const chatBox     = document.getElementById('chat-box');
const userInput   = document.getElementById('user-input');
const sendBtn     = document.getElementById('send-btn');
const stopBtn     = document.getElementById('stop-btn');
const resetBtn     = document.getElementById('reset-btn');
// Custom dropdowns (no native <select>)
const modelMenu  = document.getElementById('model-menu');
const modelLabel = document.getElementById('model-label');
const modelBtn   = document.getElementById('model-btn');
const langMenu   = document.getElementById('lang-menu');
const langLabel  = document.getElementById('lang-label');
const langBtn    = document.getElementById('lang-btn');
let selectedModel    = '';
let selectedLanguage = 'Thai';

const API_URL      = (window.location.protocol === 'file:' || window.location.hostname === 'localhost') ? 'http://127.0.0.1:8000' : window.location.origin;
const API_URL_CHAT = API_URL + '/chat';

let currentSessionId = '';
let chatHistory      = [];
let currentRemaining = 4;
let currentAbortController = null;

// ── i18n & Language ─────────────────────────────────────────
function updateLanguage() {
    const t = i18n[selectedLanguage];
    document.getElementById('app-title').textContent = t.appTitle;
    document.getElementById('user-input').placeholder = t.inputPlaceholder;
    document.getElementById('footer-text').textContent = t.footerText;
    
    const welcomeText = document.getElementById('welcome-text');
    if (welcomeText) welcomeText.innerHTML = t.welcomeMessage;
    
    updateModeUI();
}

// ── Session ─────────────────────────────────────────────────
function loadSession() {
    currentSessionId = localStorage.getItem('perfume_session_id');
    if (!currentSessionId) {
        currentSessionId = crypto.randomUUID?.() ?? Date.now().toString();
        localStorage.setItem('perfume_session_id', currentSessionId);
    }

    const saved = localStorage.getItem('chatHistory');
    if (!saved) return;

    chatHistory = JSON.parse(saved);
    if (chatHistory.length > 0) {
        document.getElementById('welcome-message')?.remove();
        chatHistory.forEach(msg => appendMessage(msg.role === 'user' ? 'user' : 'ai', msg.content));
        scrollToBottom();
    }
}



async function fetchLimitStatus() {
    try {
        const response = await fetch(API_URL + '/limit-status');
        if (response.headers.has('x-ratelimit-remaining')) {
            currentRemaining = response.headers.get('x-ratelimit-remaining');
            updateModeUI();
        }
    } catch (e) {
        console.error("Failed to fetch limit", e);
    }
}


function updateModeUI() {
    const modeText = document.getElementById('mode-text');
    const modePulse = document.getElementById('mode-pulse');
    const modeIndicator = document.getElementById('mode-indicator');
    const t = i18n[selectedLanguage];
    
    // Always show Demo Version
    modeText.textContent = t.limitRemaining.replace('{remaining}', currentRemaining);
    modePulse.style.backgroundColor = "#ff7675";
    modePulse.style.boxShadow = "0 0 10px #ff7675";
    modeIndicator.style.color = "rgba(255, 255, 255, 0.7)";
}

let persistScheduled = false;
function persistChatHistory() {
    if (persistScheduled) return;
    persistScheduled = true;
    (window.requestIdleCallback ?? (cb => setTimeout(cb, 50)))(() => {
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        persistScheduled = false;
    });
}

// ── UI helpers ──────────────────────────────────────────────
function renderText(text) {
    return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function appendMessage(sender, text, isHtml = false) {
    const wrap = document.createElement('div');
    wrap.className = `message ${sender}-message`;
    wrap.innerHTML = sender === 'user'
        ? `<div class="avatar user-avatar"><i class="fa-solid fa-user"></i></div>`
        : `<div class="avatar ai-avatar"><i class="fa-solid fa-flask"></i></div>`;

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = isHtml ? text : renderText(text);

    wrap.appendChild(content);
    chatBox.appendChild(wrap);
    scrollToBottom();
    return wrap;
}

function showTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message ai-message';
    el.id = 'typing-indicator';
    el.innerHTML = `
        <div class="avatar ai-avatar"><i class="fa-solid fa-flask"></i></div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>`;
    chatBox.appendChild(el);
    scrollToBottom();
}

function removeTypingIndicator() {
    document.getElementById('typing-indicator')?.remove();
}

let scrollScheduled = false;
function scrollToBottom() {
    if (scrollScheduled) return;
    scrollScheduled = true;
    requestAnimationFrame(() => {
        chatBox.scrollTop = chatBox.scrollHeight;
        scrollScheduled = false;
    });
}

// ── Models ──────────────────────────────────────────────────
async function fetchModels() {
    try {
        const data = await fetch(API_URL + '/models').then(r => r.json());
        modelMenu.innerHTML = '';
        data.models.forEach((m, i) => {
            const opt = document.createElement('div');
            opt.className = 'custom-select-option' + (i === 0 ? ' selected' : '');
            opt.dataset.value = m.id;
            opt.textContent = m.name;
            if (i === 0) { selectedModel = m.id; modelLabel.textContent = m.name; }
            opt.addEventListener('click', () => {
                selectedModel = m.id;
                modelLabel.textContent = m.name;
                modelMenu.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
                opt.classList.add('selected');
                modelMenu.classList.remove('open');
            });
            modelMenu.appendChild(opt);
        });
    } catch {
        modelLabel.textContent = 'Failed to load';
    }
}

// ── Send ─────────────────────────────────────────────────────
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    const model    = selectedModel;
    const language = selectedLanguage;

    // Clear input immediately (Paint frame 1)
    document.getElementById('welcome-message')?.remove();
    userInput.value    = '';
    sendBtn.style.display = 'none';
    stopBtn.style.display = 'flex';

    currentAbortController = new AbortController();

    // Yield to browser so it can paint the cleared input before building DOM
    await new Promise(resolve => requestAnimationFrame(resolve));

    // Render UI 
    appendMessage('user', text);
    showTypingIndicator();

    const historyToSend = [...chatHistory];

    try {
        const headers = { 'Content-Type': 'application/json' };

        const response = await fetch(API_URL_CHAT, {
            method : 'POST',
            headers,
            body   : JSON.stringify({
                message     : text,
                session_id  : currentSessionId,
                model,
                language,
                chat_history: historyToSend 
            }),
            signal: currentAbortController.signal
        });

        const data = await response.json();
        removeTypingIndicator();

        if (response.headers.has('x-ratelimit-remaining')) {
            currentRemaining = response.headers.get('x-ratelimit-remaining');
            updateModeUI();
        }

        if (response.status === 429 || (data && data.is_error)) {
            const t = i18n[selectedLanguage];
            const errMsg = response.status === 429 
                ? t.limitReached.replace('[Link]', `<a href="#" style="color:#a29bfe;text-decoration:underline;">GitHub</a>`)
                : data.reply;
            appendMessage('ai', errMsg, response.status === 429);
        } else {
            appendMessage('ai', data.reply);
            chatHistory.push({ role: 'user', content: text });
            chatHistory.push({ role: 'assistant', content: data.reply });
            persistChatHistory();
        }

    } catch (err) {
        console.error(err);
        removeTypingIndicator();
        if (err.name === 'AbortError') {
            appendMessage('ai', 'คุณได้ยกเลิกการรอคำตอบแล้ว');
        } else {
            const t = i18n[selectedLanguage];
            appendMessage('ai', t.connError);
        }
    } finally {
        sendBtn.style.display = 'flex';
        stopBtn.style.display = 'none';
        userInput.focus();
        currentAbortController = null;
    }
}

// ── Event Listeners ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => { 
    updateLanguage();
    fetchModels(); 
    loadSession(); 
    fetchLimitStatus();
});

sendBtn.addEventListener('click', () => setTimeout(sendMessage, 0));
stopBtn.addEventListener('click', () => {
    if (currentAbortController) {
        currentAbortController.abort();
    }
});
userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); setTimeout(sendMessage, 0); }
});

// Custom dropdown toggles
modelBtn.addEventListener('click', e => { e.stopPropagation(); modelMenu.classList.toggle('open'); langMenu.classList.remove('open'); });
langBtn.addEventListener('click',  e => { e.stopPropagation(); langMenu.classList.toggle('open');  modelMenu.classList.remove('open'); });
langMenu.querySelectorAll('.custom-select-option').forEach(opt => {
    opt.addEventListener('click', () => {
        selectedLanguage = opt.dataset.value;
        langLabel.textContent = opt.textContent;
        langMenu.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
        langMenu.classList.remove('open');
        updateLanguage();
    });
});
document.addEventListener('click', () => { modelMenu.classList.remove('open'); langMenu.classList.remove('open'); });

resetBtn.addEventListener('click', () => {
    const t = i18n[selectedLanguage];
    if (confirm(t.confirmReset)) {
        localStorage.removeItem('chatHistory');
        location.reload();
    }
});