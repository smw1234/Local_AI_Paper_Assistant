const STORAGE_KEY = "thesis-assistant-conversations-v1";

const historyList = document.getElementById("historyList");
const messageArea = document.getElementById("messageArea");
const titleElement = document.getElementById("conversationTitle");
const conversationIdElement = document.getElementById("conversationId");
const input = document.getElementById("questionInput");
const form = document.getElementById("chatForm");
const sendButton = document.getElementById("sendButton");
const newChatButton = document.getElementById("newChatButton");

let conversations = loadConversations();
let activeConversationId = null;
let isSending = false;

function loadConversations() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

function saveConversations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

function createId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function activeConversation() {
  return conversations.find((conversation) => conversation.id === activeConversationId) || null;
}

function startNewConversation() {
  const current = activeConversation();
  if (current && !window.confirm("开启新对话？当前对话会保留在历史记录中。")) {
    return;
  }
  // 这里只显示前端草稿；第一次发送问题时才在后端创建并保存会话。
  activeConversationId = null;
  titleElement.textContent = "新对话";
  conversationIdElement.textContent = "";
  renderMessages([]);
  renderHistory();
  input.focus();
}

async function openConversation(id) {
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    const result = await response.json();
    if (!response.ok || result.code !== 200) {
      throw new Error(result.message || `HTTP ${response.status}`);
    }
    let conversation = conversations.find((item) => item.id === id);
    if (!conversation) {
      conversation = { id, title: id, messages: [], updatedAt: Date.now() };
      conversations.unshift(conversation);
    }
    conversation.messages = result.data?.messages || [];
    activeConversationId = id;
    saveConversations();
    titleElement.textContent = conversation.title;
    conversationIdElement.textContent = id;
    renderMessages(conversation.messages);
    renderHistory();
  } catch (error) {
    console.error("查询历史记录失败:", error);
    window.alert(`查询历史记录失败：${error.message}`);
  }
}

function deleteConversation(id) {
  const conversation = conversations.find((item) => item.id === id);
  if (!conversation) return;
  if (!window.confirm(`确定删除会话“${conversation.title}”吗？`)) return;

  fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" })
    .then(async (response) => {
      const result = await response.json();
      if (!response.ok || result.code !== 200) throw new Error(result.message || "删除失败");
      conversations = conversations.filter((item) => item.id !== id);
      if (activeConversationId === id) activeConversationId = null;
      saveConversations();
      if (conversations.length) await openConversation(conversations[0].id);
      else await startNewConversation();
      renderHistory();
    })
    .catch((error) => window.alert(`删除会话失败：${error.message}`));
}

function renderHistory() {
  historyList.replaceChildren();

  conversations
    .slice()
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .forEach((conversation) => {
      const row = document.createElement("div");
      row.className = `history-item${conversation.id === activeConversationId ? " active" : ""}`;

      const selectButton = document.createElement("button");
      selectButton.type = "button";
      selectButton.className = "history-select";
      selectButton.textContent = conversation.title;
      selectButton.title = conversation.title;
      selectButton.addEventListener("click", () => openConversation(conversation.id));

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "delete-button";
      deleteButton.setAttribute("aria-label", `删除会话：${conversation.title}`);
      deleteButton.title = "删除会话";
      deleteButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3m3 0-.8 13H6.8L6 7m4 4v5m4-5v5"/></svg>';
      deleteButton.addEventListener("click", () => deleteConversation(conversation.id));

      row.append(selectButton, deleteButton);
      historyList.append(row);
    });
}

function renderMessages(messages) {
  messageArea.replaceChildren();

  if (!messages.length) {
    const welcome = document.createElement("div");
    welcome.className = "welcome-card";
    welcome.innerHTML = '<div class="welcome-mark">T</div><h3>开始阅读你的论文</h3><p>提出问题，我会根据已导入的论文内容为你查找答案。</p>';
    messageArea.append(welcome);
    return;
  }

  messages.forEach((message) => appendMessage(message.role, message.content));
  messageArea.scrollTop = messageArea.scrollHeight;
}

function appendMessage(role, content, extraClass = "") {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}${extraClass ? ` ${extraClass}` : ""}`;
  bubble.textContent = content;
  messageArea.append(bubble);
  messageArea.scrollTop = messageArea.scrollHeight;
  return bubble;
}

function setTitleFromQuestion(conversation, question) {
  if (conversation.messages.length === 0) {
    conversation.title = question.length > 26 ? `${question.slice(0, 26)}…` : question;
    titleElement.textContent = conversation.title;
  }
}

async function askBackend(question, conversationId) {
  // TODO: 将 URL 和请求字段改成你的 FastAPI 接口定义。
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  });

  if (!response.ok) {
    throw new Error(`服务返回错误：${response.status}`);
  }

  const data = await response.json();
  //console.log("聊天接口返回：", data, "data.data：", data.data);
  return data.data;
}

async function sendQuestion(question) {
  let conversation = activeConversation();
  if (!conversation) {
    try {
      const response = await fetch("/api/sessions", { method: "POST" });
      const result = await response.json();
      if (!response.ok || result.code !== 200 || !result.data) {
        throw new Error(result.message || `HTTP ${response.status}`);
      }
      conversation = {
        id: result.data,
        title: "新对话",
        messages: [],
        updatedAt: Date.now(),
      };
      conversations.unshift(conversation);
      activeConversationId = conversation.id;
      conversationIdElement.textContent = conversation.id;
    } catch (error) {
      window.alert(`创建会话失败：${error.message}`);
      return;
    }
  }

  setTitleFromQuestion(conversation, question);
  conversation.messages.push({ role: "user", content: question, createdAt: Date.now() });
  renderMessages(conversation.messages);
  renderHistory();
  saveConversations();

  isSending = true;
  sendButton.disabled = true;
  const pendingBubble = appendMessage("assistant", "正在查阅论文…", "pending");

  try {
    const answer = await askBackend(question, conversation.id);
    pendingBubble.remove();
    conversation.messages.push({ role: "assistant", content: answer || "接口没有返回 answer 字段。", createdAt: Date.now() });
  } catch (error) {
    pendingBubble.remove();
    const errorMessage = `暂时无法连接问答服务：${error.message}`;
    conversation.messages.push({ role: "assistant", content: errorMessage, createdAt: Date.now() });
    window.alert(errorMessage);
  } finally {
    conversation.updatedAt = Date.now();
    renderMessages(conversation.messages);
    renderHistory();
    saveConversations();
    isSending = false;
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || isSending) return;
  input.value = "";
  input.style.height = "auto";
  sendQuestion(question);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

newChatButton.addEventListener("click", startNewConversation);

async function initializeSessions() {
  try {
    const response = await fetch("/api/sessions");
    const result = await response.json();
    if (!response.ok || result.code !== 200 || !Array.isArray(result.data)) {
      throw new Error(result.message || `HTTP ${response.status}`);
    }
    const ids = result.data;
    const localById = new Map(conversations.map((item) => [item.id, item]));
    conversations = ids.map((id) => localById.get(id) || {
      id, title: id, messages: [], updatedAt: Date.now(),
    });
    saveConversations();
    renderHistory();
    // 不自动打开历史会话；进入页面始终显示未保存的新对话草稿。
    activeConversationId = null;
    titleElement.textContent = "新对话";
    conversationIdElement.textContent = "";
    renderMessages([]);
    renderHistory();
  } catch (error) {
    console.error("查询历史记录失败:", error);
    window.alert(`查询历史记录失败：${error.message}`);
    renderHistory();
    // 与参考前端一致：历史查询失败后仍尝试创建一个新会话，
    // 因此 Network 面板会分别显示 GET 和 POST /api/sessions。
    activeConversationId = null;
    titleElement.textContent = "新对话";
    conversationIdElement.textContent = "";
    renderMessages([]);
    renderHistory();
  }
}

initializeSessions();
