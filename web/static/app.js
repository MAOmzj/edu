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
  studentId: getOrCreateId(storageKeys.student, "student"),
  conversationId: getOrCreateId(storageKeys.conversation, "thread"),
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
  clearButton: document.querySelector("#clearButton"),
  forgetButton: document.querySelector("#forgetButton"),
  serviceStatus: document.querySelector("#serviceStatus"),
};

let historyReady;

function setSubject(subject) {
  state.subject = subject;
  document.querySelectorAll(".subject-button").forEach((button) => {
    const selected = button.dataset.subject === subject;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  elements.chatTitle.textContent =
    subject === "综合" ? "综合知识问答" : `${subject}知识问答`;
}

function hideWelcome() {
  if (elements.welcome) {
    elements.welcome.remove();
    elements.welcome = null;
  }
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
    bubble.textContent = content;
  }

  row.append(avatar, bubble);
  elements.messages.appendChild(row);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return row;
}

function setBusy(busy) {
  state.busy = busy;
  elements.sendButton.disabled = busy;
  elements.questionInput.disabled = busy;
  elements.sendButton.querySelector("span:first-child").textContent = busy
    ? "正在思考"
    : "发送问题";
}

function autoResize() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(
    elements.questionInput.scrollHeight,
    150,
  )}px`;
}

async function sendQuestion(question) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion || state.busy) return;

  await historyReady;

  const gradeValue = elements.gradeSelect.value;
  createMessage("user", cleanQuestion);
  const thinkingRow = createMessage("assistant", "", { thinking: true });
  elements.questionInput.value = "";
  autoResize();
  setBusy(true);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: cleanQuestion,
        subject: state.subject,
        grade: gradeValue ? Number(gradeValue) : null,
        student_id: state.studentId,
        thread_id: state.conversationId,
      }),
    });
    const data = await response.json();
    thinkingRow.remove();
    if (!response.ok) {
      throw new Error(data.detail || "回答失败，请稍后重试。");
    }
    createMessage("assistant", data.answer);
  } catch (error) {
    thinkingRow.remove();
    createMessage(
      "assistant",
      error instanceof Error ? error.message : "网络连接失败，请检查服务是否启动。",
      { error: true },
    );
  } finally {
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

async function loadHistory() {
  try {
    const url =
      `/api/history/${encodeURIComponent(state.conversationId)}` +
      `?student_id=${encodeURIComponent(state.studentId)}`;
    const response = await fetch(url);
    if (!response.ok) return;
    const data = await response.json();
    for (const message of data.messages || []) {
      createMessage(message.role, message.content);
    }
  } catch {
    // 历史加载失败不会阻止学生继续发起新问题。
  }
}

elements.subjectGrid.addEventListener("click", (event) => {
  const button = event.target.closest(".subject-button");
  if (button) setSubject(button.dataset.subject);
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    setSubject(button.dataset.subject);
    elements.questionInput.value = button.dataset.example;
    autoResize();
    elements.questionInput.focus();
  });
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

elements.clearButton.addEventListener("click", () => {
  localStorage.setItem(
    storageKeys.conversation,
    createAnonymousId("thread"),
  );
  window.location.reload();
});

elements.forgetButton.addEventListener("click", async () => {
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
checkService();
historyReady = loadHistory();
elements.questionInput.focus();
