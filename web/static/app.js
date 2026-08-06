const storageKeys = {
  student: "education_student_id",
  conversation: "education_conversation_id",
};

function createAnonymousId(prefix) {
  const randomPart = globalThis.crypto?.randomUUID
    ? globalThis.crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${randomPart}`;
}

function getOrCreateId(key, prefix) {
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const created = createAnonymousId(prefix);
  localStorage.setItem(key, created);
  return created;
}

const state = {
  subject: "综合",
  busy: false,
  loadingHistory: false,
  historyRequestNumber: 0,
  studentId: getOrCreateId(storageKeys.student, "student"),
  conversationId: getOrCreateId(storageKeys.conversation, "thread"),
  conversationTitle: "新对话",
  conversations: [],
};

const elements = {
  askForm: document.querySelector("#askForm"),
  questionInput: document.querySelector("#questionInput"),
  sendButton: document.querySelector("#sendButton"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  subjectGrid: document.querySelector("#subjectGrid"),
  gradeSelect: document.querySelector("#gradeSelect"),
  chatTitle: document.querySelector("#chatTitle"),
  chatMeta: document.querySelector("#chatMeta"),
  clearButton: document.querySelector("#clearButton"),
  newConversationButton: document.querySelector("#newConversationButton"),
  conversationList: document.querySelector("#conversationList"),
  conversationStatus: document.querySelector("#conversationStatus"),
  forgetButton: document.querySelector("#forgetButton"),
  serviceStatus: document.querySelector("#serviceStatus"),
};

let historyReady;

function currentGradeText() {
  const grade = elements.gradeSelect.value;
  return grade ? `小学${grade}年级` : "未指定年级";
}

function updateChatHeader() {
  elements.chatTitle.textContent = state.conversationTitle || "新对话";
  elements.chatMeta.textContent = `${state.subject} · ${currentGradeText()}`;
}

function setSubject(subject) {
  state.subject = subject || "综合";
  document.querySelectorAll(".subject-button").forEach((button) => {
    const selected = button.dataset.subject === state.subject;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  updateChatHeader();
}

function hideWelcome() {
  elements.welcome.hidden = true;
}

function showWelcome() {
  elements.welcome.hidden = false;
}

function clearMessages() {
  elements.messages.querySelectorAll(".message-row").forEach((row) => row.remove());
  showWelcome();
  elements.messages.scrollTop = 0;
}

function createMessage(role, content, options = {}) {
  hideWelcome();
  const row = document.createElement("div");
  row.className = `message-row ${role}${options.error ? " error" : ""}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "我" : "学";
  avatar.setAttribute("aria-hidden", "true");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (options.thinking) {
    bubble.setAttribute("aria-label", "正在思考");
    const indicator = document.createElement("span");
    indicator.className = "thinking";
    indicator.innerHTML = "<span></span><span></span><span></span>";
    bubble.appendChild(indicator);
  } else {
    // 使用 textContent 展示服务端内容，避免答案中的 HTML 被浏览器执行。
    bubble.textContent = content;
  }

  row.append(avatar, bubble);
  elements.messages.appendChild(row);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return row;
}

function beginStreamingAnswer(row) {
  const bubble = row.querySelector(".bubble");
  bubble.replaceChildren();
  bubble.removeAttribute("aria-label");
  bubble.classList.add("streaming");
  return bubble;
}

function parseServerSentEvent(block) {
  let eventName = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  return { eventName, data: dataLines.join("\n") };
}

async function readFinalAnswerStream(response, thinkingRow) {
  if (!response.body) {
    throw new Error("当前浏览器不支持流式回答，请升级浏览器后重试。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answer = "";
  let answerBubble = null;
  let completed = false;

  function handleEvent(block) {
    if (!block.trim()) return;
    const event = parseServerSentEvent(block);
    const payload = event.data ? JSON.parse(event.data) : {};

    if (event.eventName === "answer_delta") {
      if (!answerBubble) answerBubble = beginStreamingAnswer(thinkingRow);
      const text = String(payload.text || "");
      answer += text;
      answerBubble.textContent += text;
      elements.messages.scrollTop = elements.messages.scrollHeight;
    } else if (event.eventName === "done") {
      completed = true;
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    // SSE 允许 CRLF；统一成 \n 后，以空行作为一个事件的结束标记。
    buffer = buffer.replace(/\r\n/g, "\n");
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    blocks.forEach(handleEvent);

    if (done) {
      if (buffer.trim()) handleEvent(buffer);
      break;
    }
  }

  if (answerBubble) answerBubble.classList.remove("streaming");
  if (!completed || !answer) {
    throw new Error("流式回答没有完整结束，请稍后重试。");
  }
  return answer;
}

function interfaceLocked() {
  return state.busy || state.loadingHistory;
}

function updateControlState() {
  const locked = interfaceLocked();
  elements.sendButton.disabled = locked;
  elements.questionInput.disabled = locked;
  elements.clearButton.disabled = locked;
  elements.newConversationButton.disabled = locked;
  elements.conversationList
    .querySelectorAll("button")
    .forEach((button) => {
      button.disabled = locked;
    });

  elements.sendButton.querySelector("span:first-child").textContent = state.busy
    ? "正在思考"
    : state.loadingHistory
      ? "正在载入"
      : "发送问题";
}

function setBusy(busy) {
  state.busy = busy;
  updateControlState();
}

function setHistoryLoading(loading) {
  state.loadingHistory = loading;
  updateControlState();
}

function autoResize() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(
    elements.questionInput.scrollHeight,
    150,
  )}px`;
}

function formatConversationTime(value) {
  if (!value) return "刚刚创建";
  const normalized = value.includes("T") ? value : `${value.replace(" ", "T")}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return "最近使用";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function getVisibleConversations() {
  const conversations = [...state.conversations];
  const currentExists = conversations.some(
    (item) => item.thread_id === state.conversationId,
  );
  if (!currentExists) {
    conversations.unshift({
      thread_id: state.conversationId,
      title: state.conversationTitle || "新对话",
      subject: state.subject,
      grade: elements.gradeSelect.value
        ? Number(elements.gradeSelect.value)
        : null,
      updated_at: null,
      isDraft: true,
    });
  }
  return conversations;
}

function createConversationItem(conversation) {
  const active = conversation.thread_id === state.conversationId;
  const item = document.createElement("article");
  item.className = `conversation-item${active ? " active" : ""}`;
  item.setAttribute("role", "listitem");

  const openButton = document.createElement("button");
  openButton.className = "conversation-open-button";
  openButton.type = "button";
  openButton.dataset.openThread = conversation.thread_id;
  openButton.setAttribute("aria-current", active ? "true" : "false");

  const title = document.createElement("strong");
  title.textContent = conversation.title || "未命名对话";

  const meta = document.createElement("span");
  const gradeText = conversation.grade ? `${conversation.grade}年级` : "未定年级";
  meta.textContent = `${conversation.subject || "综合"} · ${gradeText} · ${formatConversationTime(
    conversation.updated_at,
  )}`;
  openButton.append(title, meta);
  item.appendChild(openButton);

  // 尚未向后端提问的新窗口没有数据库记录，因此不显示删除按钮。
  if (!conversation.isDraft) {
    const deleteButton = document.createElement("button");
    deleteButton.className = "conversation-delete-button";
    deleteButton.type = "button";
    deleteButton.dataset.deleteThread = conversation.thread_id;
    deleteButton.dataset.conversationTitle = conversation.title || "这个对话";
    deleteButton.textContent = "×";
    deleteButton.setAttribute(
      "aria-label",
      `删除会话：${conversation.title || "未命名对话"}`,
    );
    item.appendChild(deleteButton);
  }
  return item;
}

function renderConversationList() {
  elements.conversationList.replaceChildren();
  const conversations = getVisibleConversations();
  conversations.forEach((conversation) => {
    elements.conversationList.appendChild(createConversationItem(conversation));
  });

  const savedCount = state.conversations.length;
  elements.conversationStatus.textContent = savedCount
    ? `已保存 ${savedCount} 个会话`
    : "还没有历史会话，先问一个问题吧。";
  elements.conversationStatus.classList.toggle("error", false);
  updateControlState();
}

async function refreshConversations({ adoptCurrentMetadata = false } = {}) {
  const url = `/api/conversations?student_id=${encodeURIComponent(state.studentId)}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error("会话列表读取失败");
  const data = await response.json();
  state.conversations = Array.isArray(data.conversations) ? data.conversations : [];

  if (adoptCurrentMetadata) {
    const current = state.conversations.find(
      (item) => item.thread_id === state.conversationId,
    );
    if (current) {
      state.conversationTitle = current.title || "未命名对话";
      if (current.subject) setSubject(current.subject);
      elements.gradeSelect.value = current.grade ? String(current.grade) : "";
      updateChatHeader();
    }
  }
  renderConversationList();
}

async function loadHistory(threadId) {
  const requestNumber = ++state.historyRequestNumber;
  setHistoryLoading(true);
  clearMessages();

  try {
    const url =
      `/api/history/${encodeURIComponent(threadId)}` +
      `?student_id=${encodeURIComponent(state.studentId)}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error("历史记录读取失败");
    const data = await response.json();

    // 如果用户已经打开另一个窗口，旧请求即使稍晚返回也不能覆盖新窗口。
    if (requestNumber !== state.historyRequestNumber) return;
    clearMessages();
    for (const message of data.messages || []) {
      createMessage(message.role, message.content);
    }
  } catch {
    if (requestNumber === state.historyRequestNumber) {
      clearMessages();
      createMessage("assistant", "这个会话暂时无法读取，请稍后再试。", {
        error: true,
      });
    }
  } finally {
    if (requestNumber === state.historyRequestNumber) {
      setHistoryLoading(false);
      elements.questionInput.focus();
    }
  }
}

function startNewConversation() {
  if (interfaceLocked()) return;

  // 新 thread_id 就是一个全新的短期记忆窗口，student_id 保持不变以共享长期记忆。
  state.historyRequestNumber += 1;
  state.conversationId = createAnonymousId("thread");
  state.conversationTitle = "新对话";
  localStorage.setItem(storageKeys.conversation, state.conversationId);
  clearMessages();
  updateChatHeader();
  renderConversationList();
  historyReady = Promise.resolve();
  elements.questionInput.focus();
}

async function switchConversation(threadId) {
  if (interfaceLocked() || threadId === state.conversationId) return;
  const conversation = state.conversations.find((item) => item.thread_id === threadId);
  if (!conversation) return;

  state.conversationId = threadId;
  state.conversationTitle = conversation.title || "未命名对话";
  localStorage.setItem(storageKeys.conversation, threadId);
  setSubject(conversation.subject || "综合");
  elements.gradeSelect.value = conversation.grade ? String(conversation.grade) : "";
  updateChatHeader();
  renderConversationList();
  historyReady = loadHistory(threadId);
  await historyReady;
}

async function deleteConversation(threadId, title) {
  if (interfaceLocked()) return;
  const confirmed = window.confirm(`确定删除“${title}”吗？删除后无法恢复。`);
  if (!confirmed) return;

  setHistoryLoading(true);
  try {
    const url =
      `/api/conversations/${encodeURIComponent(threadId)}` +
      `?student_id=${encodeURIComponent(state.studentId)}`;
    const response = await fetch(url, { method: "DELETE" });
    if (!response.ok) throw new Error();

    state.conversations = state.conversations.filter(
      (item) => item.thread_id !== threadId,
    );

    if (threadId === state.conversationId) {
      const nextConversation = state.conversations[0];
      if (nextConversation) {
        state.conversationId = nextConversation.thread_id;
        state.conversationTitle = nextConversation.title || "未命名对话";
        localStorage.setItem(storageKeys.conversation, state.conversationId);
        setSubject(nextConversation.subject || "综合");
        elements.gradeSelect.value = nextConversation.grade
          ? String(nextConversation.grade)
          : "";
        updateChatHeader();
        renderConversationList();
        setHistoryLoading(false);
        historyReady = loadHistory(state.conversationId);
        await historyReady;
        return;
      }

      setHistoryLoading(false);
      startNewConversation();
      return;
    }

    renderConversationList();
  } catch {
    window.alert("会话删除失败，请确认服务仍在运行后重试。");
  } finally {
    setHistoryLoading(false);
  }
}

async function sendQuestion(question) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion || interfaceLocked()) return;

  await historyReady;
  if (interfaceLocked()) return;

  const gradeValue = elements.gradeSelect.value;
  const requestConversationId = state.conversationId;
  if (state.conversationTitle === "新对话") {
    state.conversationTitle =
      cleanQuestion.length > 28 ? `${cleanQuestion.slice(0, 28)}…` : cleanQuestion;
    updateChatHeader();
    renderConversationList();
  }

  createMessage("user", cleanQuestion);
  const thinkingRow = createMessage("assistant", "", { thinking: true });
  elements.questionInput.value = "";
  autoResize();
  setBusy(true);

  try {
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: cleanQuestion,
        subject: state.subject,
        grade: gradeValue ? Number(gradeValue) : null,
        student_id: state.studentId,
        thread_id: requestConversationId,
      }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || "回答失败，请稍后重试。");
    }
    await readFinalAnswerStream(response, thinkingRow);
  } catch (error) {
    thinkingRow.remove();
    if (state.conversationId === requestConversationId) {
      createMessage(
        "assistant",
        error instanceof Error ? error.message : "网络连接失败，请检查服务是否启动。",
        { error: true },
      );
    }
  } finally {
    try {
      await refreshConversations({ adoptCurrentMetadata: true });
    } catch {
      renderConversationList();
    }
    setBusy(false);
    elements.questionInput.focus();
  }
}

async function checkService() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    elements.serviceStatus.className = "service-status online";
    elements.serviceStatus.querySelector("span:last-child").textContent =
      "服务已连接 · 记忆已开启";
  } catch {
    elements.serviceStatus.className = "service-status offline";
    elements.serviceStatus.querySelector("span:last-child").textContent =
      "服务未连接";
  }
}

async function initializeConversations() {
  setHistoryLoading(true);
  try {
    await refreshConversations({ adoptCurrentMetadata: true });
  } catch {
    state.conversations = [];
    renderConversationList();
    elements.conversationStatus.textContent = "会话列表暂时无法读取";
    elements.conversationStatus.classList.add("error");
  } finally {
    setHistoryLoading(false);
  }
  await loadHistory(state.conversationId);
}

elements.subjectGrid.addEventListener("click", (event) => {
  const button = event.target.closest(".subject-button");
  if (button) setSubject(button.dataset.subject);
});

elements.messages.addEventListener("click", (event) => {
  const button = event.target.closest("[data-example]");
  if (!button) return;
  setSubject(button.dataset.subject);
  elements.questionInput.value = button.dataset.example;
  autoResize();
  elements.questionInput.focus();
});

elements.conversationList.addEventListener("click", (event) => {
  const deleteButton = event.target.closest("[data-delete-thread]");
  if (deleteButton) {
    deleteConversation(
      deleteButton.dataset.deleteThread,
      deleteButton.dataset.conversationTitle,
    );
    return;
  }

  const openButton = event.target.closest("[data-open-thread]");
  if (openButton) switchConversation(openButton.dataset.openThread);
});

elements.askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(elements.questionInput.value);
});

elements.questionInput.addEventListener("input", autoResize);
elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.askForm.requestSubmit();
  }
});

elements.gradeSelect.addEventListener("change", updateChatHeader);
elements.clearButton.addEventListener("click", startNewConversation);
elements.newConversationButton.addEventListener("click", startNewConversation);

elements.forgetButton.addEventListener("click", async () => {
  if (interfaceLocked()) return;
  const confirmed = window.confirm(
    "这会删除本浏览器匿名学生的全部会话和长期学习记忆，确定继续吗？",
  );
  if (!confirmed) return;

  elements.forgetButton.disabled = true;
  try {
    const response = await fetch(
      `/api/students/${encodeURIComponent(state.studentId)}/memory`,
      { method: "DELETE" },
    );
    if (!response.ok) throw new Error();
    localStorage.removeItem(storageKeys.student);
    localStorage.removeItem(storageKeys.conversation);
    window.location.reload();
  } catch {
    window.alert("记忆删除失败，请确认服务仍在运行后重试。");
    elements.forgetButton.disabled = false;
  }
});

setSubject("综合");
renderConversationList();
checkService();
historyReady = initializeConversations();
