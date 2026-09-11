import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "10mb" }));

// Helper to get GoogleGenAI client lazily
let geminiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!geminiClient) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error("GEMINI_API_KEY environment variable is missing");
    }
    geminiClient = new GoogleGenAI({ apiKey: key });
  }
  return geminiClient;
}

// Health check
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    hasBuiltinGemini: Boolean(process.env.GEMINI_API_KEY),
    timestamp: new Date().toISOString(),
  });
});

// Proxy: Fetch models from custom base URL
app.post("/api/proxy/models", async (req, res) => {
  const { baseUrl, apiKey } = req.body;

  if (!baseUrl || !baseUrl.trim()) {
    return res.status(400).json({
      error: "请先输入 API 接口地址 (Base URL)",
      models: [],
    });
  }

  if (!apiKey || !apiKey.trim()) {
    return res.status(400).json({
      error: "请先输入 API 密钥 (API Key)",
      models: [],
    });
  }

  try {
    let cleanUrl = (baseUrl || "").trim().replace(/\/+$/, "");
    if (!cleanUrl.endsWith("/v1") && !cleanUrl.endsWith("/models")) {
      cleanUrl += "/v1";
    }
    const targetUrl = cleanUrl.endsWith("/models") ? cleanUrl : `${cleanUrl}/models`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey.trim()}`;
    }

    const abortCtrl = new AbortController();
    const timeout = setTimeout(() => abortCtrl.abort(), 12000);

    const remoteRes = await fetch(targetUrl, {
      method: "GET",
      headers,
      signal: abortCtrl.signal,
    });
    clearTimeout(timeout);

    if (!remoteRes.ok) {
      const errorText = await remoteRes.text().catch(() => "");
      return res.status(remoteRes.status).json({
        error: `API 响应状态 ${remoteRes.status}: ${errorText.slice(0, 200) || remoteRes.statusText}`,
      });
    }

    const data = await remoteRes.json();
    let modelList: Array<{ id: string; name: string }> = [];

    if (Array.isArray(data)) {
      modelList = data.map((item: any) => ({
        id: item.id || item.name || String(item),
        name: item.name || item.id || String(item),
      }));
    } else if (Array.isArray(data.data)) {
      modelList = data.data.map((item: any) => ({
        id: item.id || item.name || String(item),
        name: item.id || item.name || String(item),
      }));
    } else if (Array.isArray(data.models)) {
      modelList = data.models.map((item: any) => ({
        id: item.name || item.id,
        name: item.displayName || item.name || item.id,
      }));
    }

    // Always append built-in gemini as option if available
    return res.json({
      models: modelList,
      source: "custom-api",
      rawCount: modelList.length,
    });
  } catch (err: any) {
    console.error("Error fetching models:", err);
    return res.status(500).json({
      error: err.name === "AbortError" ? "请求超时，请检查API地址与网络连通性" : (err.message || "拉取模型列表失败"),
    });
  }
});

// Proxy: Chat Completions (with SSE streaming or non-streaming)
app.post("/api/proxy/chat", async (req, res) => {
  const { baseUrl, apiKey, model, messages, stream = false, temperature = 0.7 } = req.body;

  if (!baseUrl || !baseUrl.trim() || !apiKey || !apiKey.trim()) {
    return res.status(400).json({
      error: "当前未配置 API 接口地址或密钥，请先在【API 配置】中设置并拉取选择 AI 模型。",
    });
  }

  if (!model || !model.trim()) {
    return res.status(400).json({
      error: "当前未选择 AI 模型，请先在【API 配置】中拉取并选定模型。",
    });
  }

  // Standard OpenAI-compatible proxy call
  try {
    let cleanUrl = (baseUrl || "").trim().replace(/\/+$/, "");
    if (!cleanUrl.endsWith("/v1") && !cleanUrl.endsWith("/chat/completions")) {
      cleanUrl += "/v1";
    }
    const targetUrl = cleanUrl.endsWith("/chat/completions") ? cleanUrl : `${cleanUrl}/chat/completions`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey.trim()}`;
    }

    const payload: any = {
      model: model || "gpt-3.5-turbo",
      messages,
      temperature: Number(temperature) || 0.7,
      stream: Boolean(stream),
    };

    if (stream) {
      const remoteRes = await fetch(targetUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      if (!remoteRes.ok) {
        const errBody = await remoteRes.text().catch(() => "");
        return res.status(remoteRes.status).json({
          error: `API 报错 (${remoteRes.status}): ${errBody.slice(0, 300) || remoteRes.statusText}`,
        });
      }

      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      if (!remoteRes.body) {
        throw new Error("No readable stream received from upstream API");
      }

      const reader = remoteRes.body.getReader();
      const decoder = new TextDecoder("utf-8");

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunkStr = decoder.decode(value, { stream: true });
          res.write(chunkStr);
        }
      } catch (streamErr: any) {
        console.error("Stream reading error:", streamErr);
      } finally {
        res.end();
      }
    } else {
      const remoteRes = await fetch(targetUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      if (!remoteRes.ok) {
        const errBody = await remoteRes.text().catch(() => "");
        return res.status(remoteRes.status).json({
          error: `API 报错 (${remoteRes.status}): ${errBody.slice(0, 300) || remoteRes.statusText}`,
        });
      }

      const json = await remoteRes.json();
      return res.json(json);
    }
  } catch (err: any) {
    console.error("Proxy error:", err);
    if (!res.headersSent) {
      return res.status(500).json({ error: err.message || "请求自定义 API 端点失败" });
    }
    res.end();
  }
});

// Serve SillyTavern extension files directly for easy loading
app.get("/api/sillytavern/extension-code", (req, res) => {
  const host = req.get("host") || "localhost:3000";
  const protocol = req.protocol;
  const currentAppUrl = `${protocol}://${host}`;

  const extensionScript = `
/**
 * SillyTavern 24H French Editorial Schedule Extension
 * Loads the French Magazine 24h Schedule within SillyTavern
 */
(function() {
  const MODULE_NAME = 'french_24h_schedule';
  console.log('[French 24h Schedule] Initializing extension...');

  function getSTContext() {
    try {
      const context = SillyTavern.getContext();
      const char = context.characters?.[context.characterId] || null;
      const user = context.name2 || 'User';
      const userAvatar = context.userAvatar;
      const worldbook = context.worldInfo || null;
      const chat = context.chat || [];
      return {
        connected: true,
        charId: context.characterId,
        charName: char?.name || context.characters?.[0]?.name || 'Unknown Char',
        charPersonality: char?.personality || char?.description || '',
        charScenario: char?.scenario || '',
        userName: user,
        userDescription: context.power_user?.persona_description || '',
        worldbookEntries: (worldbook?.entries || []).slice(0, 20).map(e => ({
          key: e.key || e.comment || 'Entry',
          content: e.content
        })),
        recentMessages: chat.slice(-6).map(m => ({
          name: m.name,
          mes: m.mes,
          is_user: m.is_user
        }))
      };
    } catch(e) {
      console.warn('[French 24h Schedule] SillyTavern context query:', e);
      return { connected: false, error: e.message };
    }
  }

  // Listen to postMessage from iframe
  window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'REQ_SILLYTAVERN_CONTEXT') {
      const ctx = getSTContext();
      event.source.postMessage({
        type: 'RESP_SILLYTAVERN_CONTEXT',
        payload: ctx
      }, '*');
    }
    if (event.data && event.data.type === 'SEND_SCHEDULE_TO_CHAT') {
      try {
        const text = event.data.text;
        const sendInput = document.getElementById('send_textarea');
        if (sendInput && text) {
          sendInput.value = text;
          sendInput.dispatchEvent(new Event('input', { bubbles: true }));
        }
      } catch(e) {
        console.error('Failed to paste to SillyTavern chat', e);
      }
    }
  });

  // Add Button to SillyTavern Top Nav or Extensions Bar
  function addToolbarButton() {
    if (document.getElementById('french-schedule-btn')) return;
    const extensionsMenu = document.getElementById('extensions_settings') || document.getElementById('top-bar');
    const btn = document.createElement('div');
    btn.id = 'french-schedule-btn';
    btn.className = 'menu_button fa-solid fa-calendar-day';
    btn.title = '法式 24H 日程表 (French Schedule)';
    btn.style.cursor = 'pointer';
    btn.style.display = 'inline-flex';
    btn.style.alignItems = 'center';
    btn.style.justifyContent = 'center';
    btn.style.padding = '6px 10px';
    btn.style.margin = '2px';
    btn.innerText = ' 24H 日程';

    btn.onclick = () => {
      openScheduleModal();
    };

    if (extensionsMenu) {
      extensionsMenu.appendChild(btn);
    }
  }

  function openScheduleModal() {
    let modal = document.getElementById('french-schedule-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'french-schedule-modal';
      modal.style.position = 'fixed';
      modal.style.top = '0';
      modal.style.left = '0';
      modal.style.width = '100vw';
      modal.style.height = '100vh';
      modal.style.backgroundColor = 'rgba(0,0,0,0.6)';
      modal.style.zIndex = '99999';
      modal.style.display = 'flex';
      modal.style.alignItems = 'center';
      modal.style.justifyContent = 'center';

      const container = document.createElement('div');
      container.style.width = '95vw';
      container.style.maxWidth = '480px';
      container.style.height = '92vh';
      container.style.backgroundColor = '#F8F7F4';
      container.style.borderRadius = '16px';
      container.style.overflow = 'hidden';
      container.style.display = 'flex';
      container.style.flexDirection = 'column';
      container.style.boxShadow = '0 25px 50px -12px rgba(0,0,0,0.5)';

      const header = document.createElement('div');
      header.style.display = 'flex';
      header.style.justifyContent = 'space-between';
      header.style.alignItems = 'center';
      header.style.padding = '8px 16px';
      header.style.borderBottom = '1px solid #E5E4DE';
      header.style.backgroundColor = '#F8F7F4';
      header.innerHTML = '<span style="font-family:serif;font-weight:600;letter-spacing:1px;font-size:14px;color:#1E1E1C;">L\\'HORAIRE • 24 HEURES</span><button id="close-fr-sched" style="border:none;background:none;font-size:20px;cursor:pointer;color:#555;">&times;</button>';

      const iframe = document.createElement('iframe');
      iframe.src = '${currentAppUrl}?st_embedded=1';
      iframe.style.width = '100%';
      iframe.style.height = '100%';
      iframe.style.border = 'none';

      container.appendChild(header);
      container.appendChild(iframe);
      modal.appendChild(container);
      document.body.appendChild(modal);

      modal.querySelector('#close-fr-sched').onclick = () => {
        modal.style.display = 'none';
      };
    } else {
      modal.style.display = 'flex';
    }
  }

  // Interval hook to ensure button attaches to SillyTavern UI
  const timer = setInterval(() => {
    if (window.SillyTavern || document.getElementById('top-bar')) {
      addToolbarButton();
      clearInterval(timer);
    }
  }, 1000);
})();
`;

  res.setHeader("Content-Type", "application/javascript");
  res.send(extensionScript);
});

// Vite middleware & Static serving
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`French 24h Schedule server running on port ${PORT}`);
  });
}

startServer();
