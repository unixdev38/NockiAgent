/* Nocki Agent — chat UI (vanilla JS, no build step).
 *
 * Talks to the FastAPI backend on the same origin:
 *   GET  /health, /mcp, /sessions, /sessions/{id}, /files
 *   POST /files (multipart), /chat/stream (SSE over fetch), DELETE /sessions/{id}
 * All model/tool output is rendered with textContent or escaped HTML — never raw.
 */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const els = {
    sessions: $("#sessions"),
    files: $("#files"),
    mcp: $("#mcp"),
    thread: $("#thread"),
    messages: $("#messages"),
    input: $("#input"),
    send: $("#send"),
    composer: $("#composer"),
    chips: $("#chips"),
    chatTitle: $("#chat-title"),
    statusDot: $("#status-dot"),
    statusText: $("#status-text"),
    sidebar: $("#sidebar"),
    fileInput: $("#file-input"),
  };

  const TITLES_KEY = "nocki.sessionTitles";
  const state = {
    sessionId: null,
    busy: false,
    pending: [], // uploaded files waiting to be attached to the next message
  };

  // ---------------------------------------------------------------- helpers
  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  }

  // Minimal, safe Markdown: input is escaped before any formatting is applied.
  function renderMarkdown(src) {
    return src
      .split("```")
      .map((part, i) => {
        if (i % 2 === 1) {
          const nl = part.indexOf("\n");
          const code = nl >= 0 ? part.slice(nl + 1) : part;
          return `<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;
        }
        return renderBlocks(escapeHtml(part));
      })
      .join("");
  }

  function renderBlocks(text) {
    let html = "";
    let list = null;
    for (const line of text.split("\n")) {
      const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
      const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
      const heading = /^(#{1,3})\s+(.*)$/.exec(line);
      const item = bullet || numbered;
      if (item) {
        const tag = bullet ? "ul" : "ol";
        if (list !== tag) {
          if (list) html += `</${list}>`;
          html += `<${tag}>`;
          list = tag;
        }
        html += `<li>${inline(item[1])}</li>`;
        continue;
      }
      if (list) {
        html += `</${list}>`;
        list = null;
      }
      if (heading) {
        const level = heading[1].length + 2; // # → h3, ## → h4, ### → h5
        html += `<h${level}>${inline(heading[2])}</h${level}>`;
      } else if (line.trim() === "") {
        html += "";
      } else {
        html += `<p>${inline(line)}</p>`;
      }
    }
    if (list) html += `</${list}>`;
    return html;
  }

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch (_) { /* not JSON */ }
      throw new Error(detail);
    }
    return res.status === 204 ? null : res.json();
  }

  const titles = () => JSON.parse(localStorage.getItem(TITLES_KEY) || "{}");
  function setTitle(id, text) {
    const all = titles();
    if (!all[id]) {
      all[id] = text.slice(0, 60);
      localStorage.setItem(TITLES_KEY, JSON.stringify(all));
    }
  }
  function forgetTitle(id) {
    const all = titles();
    delete all[id];
    localStorage.setItem(TITLES_KEY, JSON.stringify(all));
  }

  // ---------------------------------------------------------------- rendering
  function scrollDown() {
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  function hideWelcome() {
    const node = document.getElementById("welcome");
    if (node) node.remove();
  }

  function showWelcome() {
    els.thread.innerHTML = "";
    const node = document.createElement("div");
    node.className = "welcome";
    node.id = "welcome";
    node.innerHTML =
      "<h1>Bonjour 👋</h1>" +
      "<p>Posez-moi une question, demandez-moi de lire un de vos documents, ou joignez un fichier avec le trombone.</p>" +
      '<div class="suggestions">' +
      '<button class="suggestion">Résume le dernier document envoyé</button>' +
      '<button class="suggestion">Liste les fichiers du workspace</button>' +
      '<button class="suggestion">Quels outils MCP sont disponibles ?</button>' +
      "</div>";
    node.querySelectorAll(".suggestion").forEach((btn) =>
      btn.addEventListener("click", () => sendMessage(btn.textContent))
    );
    els.thread.append(node);
  }

  function addMessage(role, text) {
    hideWelcome();
    const row = document.createElement("div");
    row.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "user") {
      bubble.textContent = text;
    } else {
      bubble.innerHTML = renderMarkdown(text);
    }
    if (role === "assistant") {
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = "◆";
      row.append(avatar, bubble);
    } else {
      row.append(bubble);
    }
    els.thread.append(row);
    scrollDown();
    return bubble;
  }

  function addTypingIndicator() {
    const row = document.createElement("div");
    row.className = "msg assistant";
    row.id = "typing";
    row.innerHTML = '<div class="avatar">◆</div><div class="bubble typing"><span></span><span></span><span></span></div>';
    els.thread.append(row);
    scrollDown();
  }

  function removeTypingIndicator() {
    const node = document.getElementById("typing");
    if (node) node.remove();
  }

  function addToolEvent(kind, tool, text) {
    const box = document.createElement("details");
    box.className = "tool" + (kind === "tool_result" && text.startsWith("[error]") ? " error" : "");
    const summary = document.createElement("summary");
    summary.textContent = kind === "tool_call" ? `🔧 ${text}` : `↳ résultat de ${tool}`;
    box.append(summary);
    if (kind === "tool_result") {
      const pre = document.createElement("pre");
      pre.textContent = text;
      box.append(pre);
    }
    els.thread.append(box);
    scrollDown();
  }

  function addErrorBanner(text) {
    const box = document.createElement("div");
    box.className = "tool error";
    box.textContent = `⚠ ${text}`;
    els.thread.append(box);
    scrollDown();
  }

  function renderSessions(list) {
    els.sessions.innerHTML = "";
    const known = titles();
    const sorted = [...list].sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1));
    if (!sorted.length) {
      const empty = document.createElement("li");
      empty.className = "hint";
      empty.textContent = "Aucune conversation pour l'instant.";
      els.sessions.append(empty);
    }
    for (const s of sorted) {
      const li = document.createElement("li");
      li.className = "clickable" + (s.id === state.sessionId ? " active" : "");
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = known[s.id] || `Conversation ${s.id.slice(0, 6)}`;
      name.title = s.id;
      const del = document.createElement("button");
      del.className = "del";
      del.title = "Supprimer";
      del.setAttribute("aria-label", "Supprimer la conversation");
      del.textContent = "🗑";
      del.addEventListener("click", (ev) => {
        ev.stopPropagation();
        deleteSession(s.id);
      });
      li.append(name, del);
      li.addEventListener("click", () => openSession(s.id));
      els.sessions.append(li);
    }
  }

  function renderFiles(list) {
    els.files.innerHTML = "";
    if (!list.length) {
      const empty = document.createElement("li");
      empty.className = "hint";
      empty.textContent = "Aucun document envoyé.";
      els.files.append(empty);
    }
    for (const f of list) {
      const li = document.createElement("li");
      li.className = "clickable";
      li.title = "Cliquer pour joindre à votre message";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = f.name;
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = formatSize(f.size);
      li.append(name, meta);
      li.addEventListener("click", () => attach(f));
      els.files.append(li);
    }
  }

  function renderMcp(list) {
    els.mcp.innerHTML = "";
    if (!list.length) {
      const empty = document.createElement("li");
      empty.className = "hint";
      empty.textContent = "Aucun serveur MCP configuré (voir mcp_servers.json).";
      els.mcp.append(empty);
      return;
    }
    for (const srv of list) {
      const li = document.createElement("li");
      li.title = srv.error || srv.tools.join(", ") || "";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = `${srv.name} (${srv.tools.length})`;
      const badge = document.createElement("span");
      badge.className = `badge ${srv.state}`;
      badge.textContent = srv.state;
      li.append(name, badge);
      els.mcp.append(li);
    }
  }

  function renderChips() {
    els.chips.innerHTML = "";
    state.pending.forEach((f, index) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      const label = document.createElement("span");
      label.textContent = `📄 ${f.name}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "✕";
      remove.setAttribute("aria-label", `Retirer ${f.name}`);
      remove.addEventListener("click", () => {
        state.pending.splice(index, 1);
        renderChips();
      });
      chip.append(label, remove);
      els.chips.append(chip);
    });
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} o`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
    return `${(bytes / 1024 / 1024).toFixed(1)} Mo`;
  }

  // ---------------------------------------------------------------- data loading
  async function refreshHealth() {
    try {
      const h = await api("/health");
      els.statusDot.className = "dot " + (h.llm_available ? "ok" : "ko");
      const where = h.provider === "openrouter" ? "OpenRouter (cloud)" : "Ollama (local)";
      els.statusText.textContent = `${h.model} · ${where}` + (h.llm_available ? "" : " — indisponible");
    } catch (err) {
      els.statusDot.className = "dot ko";
      els.statusText.textContent = "Serveur injoignable";
    }
  }

  async function refreshSessions() {
    renderSessions(await api("/sessions"));
  }

  async function refreshFiles() {
    renderFiles(await api("/files"));
  }

  async function refreshMcp() {
    renderMcp(await api("/mcp"));
  }

  async function refreshAll() {
    await Promise.allSettled([refreshHealth(), refreshSessions(), refreshFiles(), refreshMcp()]);
  }

  // ---------------------------------------------------------------- sessions
  function newChat() {
    state.sessionId = null;
    state.pending = [];
    renderChips();
    showWelcome();
    els.chatTitle.textContent = "Nouvelle conversation";
    els.sidebar.classList.remove("open");
    refreshSessions().catch(() => {});
    els.input.focus();
  }

  async function openSession(id) {
    try {
      const detail = await api(`/sessions/${encodeURIComponent(id)}`);
      state.sessionId = id;
      state.pending = [];
      renderChips();
      els.thread.innerHTML = "";
      for (const m of detail.messages) {
        if (m.role === "user" || m.role === "assistant") addMessage(m.role, m.content);
      }
      els.chatTitle.textContent = titles()[id] || `Conversation ${id.slice(0, 6)}`;
      els.sidebar.classList.remove("open");
      await refreshSessions();
    } catch (err) {
      addErrorBanner(`Impossible d'ouvrir la conversation : ${err.message}`);
    }
  }

  async function deleteSession(id) {
    if (!confirm("Supprimer cette conversation ?")) return;
    try {
      await api(`/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
      forgetTitle(id);
      if (state.sessionId === id) newChat();
      await refreshSessions();
    } catch (err) {
      alert(`Suppression impossible : ${err.message}`);
    }
  }

  // ---------------------------------------------------------------- uploads
  async function uploadFiles(fileList) {
    for (const file of fileList) {
      const form = new FormData();
      form.append("file", file, file.name);
      try {
        const saved = await api("/files", { method: "POST", body: form });
        attach(saved);
      } catch (err) {
        addErrorBanner(`Envoi de « ${file.name} » refusé : ${err.message}`);
      }
    }
    await refreshFiles();
  }

  // Attach a stored document: the agent reads the text sidecar when there is one.
  function attach(file) {
    if (state.pending.some((f) => f.path === file.path)) return;
    state.pending.push({ name: file.name, path: file.extracted_path || file.path });
    renderChips();
    els.input.focus();
  }

  // ---------------------------------------------------------------- chat
  function buildOutgoing(text) {
    if (!state.pending.length) return text;
    const refs = state.pending.map((f) => `- ${f.path}`).join("\n");
    state.pending = [];
    renderChips();
    return `${text}\n\nDocuments joints (lis-les avec read_file si besoin) :\n${refs}`.trim();
  }

  async function sendMessage(rawText) {
    const text = rawText.trim();
    if (!text || state.busy) return;
    const outgoing = buildOutgoing(text);
    state.busy = true;
    els.send.disabled = true;
    addMessage("user", outgoing);
    els.input.value = "";
    autoGrow();
    addTypingIndicator();

    try {
      const res = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ message: outgoing, session_id: state.sessionId }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      await readSse(res.body, (event, data) => {
        if (event === "session" && data.session_id) {
          state.sessionId = data.session_id;
          setTitle(state.sessionId, text);
          els.chatTitle.textContent = titles()[state.sessionId] || text.slice(0, 60);
        } else if (event === "tool_call" || event === "tool_result") {
          removeTypingIndicator();
          addToolEvent(event, data.tool, data.text || "");
          if (event === "tool_result") addTypingIndicator();
        } else if (event === "final") {
          removeTypingIndicator();
          addMessage("assistant", data.text || "");
        } else if (event === "error") {
          removeTypingIndicator();
          addErrorBanner(data.text || data.error || "Erreur inconnue");
        }
      });
    } catch (err) {
      removeTypingIndicator();
      addErrorBanner(`Connexion interrompue : ${err.message}`);
    } finally {
      removeTypingIndicator();
      state.busy = false;
      els.send.disabled = false;
      els.input.focus();
      refreshSessions().catch(() => {});
    }
  }

  // Parse a text/event-stream body (sse-starlette uses CRLF line endings).
  async function readSse(body, onEvent) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let match;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        const block = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const parsed = parseBlock(block);
        if (parsed) onEvent(parsed.event, parsed.data);
      }
    }
  }

  function parseBlock(block) {
    let event = "message";
    const dataLines = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
    }
    if (!dataLines.length) return null;
    let data = {};
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch (_) {
      data = { text: dataLines.join("\n") };
    }
    if (event === "done") return null;
    return { event, data };
  }

  function autoGrow() {
    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(els.input.scrollHeight, 200)}px`;
  }

  // ---------------------------------------------------------------- wiring
  els.composer.addEventListener("submit", (ev) => {
    ev.preventDefault();
    sendMessage(els.input.value);
  });
  els.input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      sendMessage(els.input.value);
    }
  });
  els.input.addEventListener("input", autoGrow);

  $("#new-chat").addEventListener("click", newChat);
  $("#attach-btn").addEventListener("click", () => els.fileInput.click());
  $("#upload-btn").addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => {
    uploadFiles(Array.from(els.fileInput.files || []));
    els.fileInput.value = "";
  });
  $("#open-sidebar").addEventListener("click", () => els.sidebar.classList.add("open"));
  $("#close-sidebar").addEventListener("click", () => els.sidebar.classList.remove("open"));

  // Drag & drop anywhere on the page.
  ["dragover", "drop"].forEach((type) =>
    document.addEventListener(type, (ev) => ev.preventDefault())
  );
  document.addEventListener("drop", (ev) => {
    if (ev.dataTransfer && ev.dataTransfer.files.length) uploadFiles(Array.from(ev.dataTransfer.files));
  });

  showWelcome();
  refreshAll();
  setInterval(refreshHealth, 30000);
  setInterval(refreshMcp, 60000);
})();
