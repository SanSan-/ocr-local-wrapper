"use strict";

function errorMessage(payload, field = "detail") {
  const message = typeof payload?.[field] === "string" ? payload[field] : "Не удалось выполнить запрос. Повторите попытку.";
  const details = [];
  if (typeof payload?.code === "string") details.push("код: " + payload.code);
  if (typeof payload?.diagnostic_id === "string") details.push("диагностика: " + payload.diagnostic_id);
  return message + (details.length ? " (" + details.join(", ") + ")" : "");
}

async function readError(response) {
  try {
    const payload = await response.json();
    let message = errorMessage(payload);
    const retryAfter = response.headers.get("Retry-After");
    if (response.status === 429 && retryAfter && /^\d+$/.test(retryAfter)) {
      message += " Повторите попытку через " + retryAfter + " с.";
    }
    return message;
  } catch { return "Не удалось выполнить запрос. Повторите попытку."; }
}

function getClipboardImage(event) {
  for (const item of Array.from(event.clipboardData?.items || [])) {
    if (item.kind !== "file" || !item.type.startsWith("image/")) continue;
    const file = item.getAsFile();
    const extensions = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/bmp": "bmp"};
    const extension = extensions[file?.type];
    if (!file || !extension) continue;
    return new File([file], "clipboard-" + Date.now() + "." + extension, {type: file.type});
  }
  return null;
}

function initializeApp() {
  const $ = (id) => document.getElementById(id);
  const dropZone = $("dropZone");
  const fileInput = $("fileInput");
  const imagePreview = $("imagePreview");
  const resultText = $("resultText");
  const recognizeBtn = $("recognizeBtn");
  const logConsole = $("logConsole");
  const settingsForm = $("modelSettings");
  const modelSelect = $("modelSelect");
  const promptInput = $("promptInput");
  const maxTokensInput = $("maxTokensInput");
  const maxPixelsInput = $("maxPixelsInput");
  const quantizationInput = $("quantizationInput");
  const forceInput = $("forceInput");
  const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
  const MAX_PROMPT_BYTES = 8 * 1024;
  const utf8 = new TextEncoder();
  const state = {uploadId: null, jobId: null, stream: null, busy: false, ready: false, models: [], modelsError: "", revision: 0};

  function setStatus(text) { $("statusText").textContent = text; }

  function appendLog(text) {
    if (!text) return;
    logConsole.textContent += text.endsWith("\n") ? text : text + "\n";
    logConsole.scrollTop = logConsole.scrollHeight;
  }

  function setBusy(busy) {
    state.busy = busy;
    recognizeBtn.disabled = busy || !state.uploadId || !state.ready;
    recognizeBtn.textContent = state.jobId ? "Распознаётся..." : "Распознать";
    for (const control of settingsForm.elements) control.disabled = busy || !state.ready;
    fileInput.disabled = busy || !state.ready;
    dropZone.disabled = busy || !state.ready;
    dropZone.setAttribute("aria-disabled", String(busy || !state.ready));
  }

  function selectedSettings() {
    return {
      model_id: modelSelect.value, prompt: promptInput.value,
      max_new_tokens: Number(maxTokensInput.value), max_pixels: Number(maxPixelsInput.value),
      quantization_enabled: quantizationInput.checked,
    };
  }

  function applyDefaults() {
    const model = state.models.find((item) => item.id === modelSelect.value);
    if (!model) return;
    promptInput.value = model.prompt;
    maxTokensInput.value = model.max_new_tokens;
    maxPixelsInput.value = model.max_pixels;
    maxPixelsInput.min = model.min_pixels;
    quantizationInput.checked = false;
    $("modelHint").textContent = model.available
      ? "Модель доступна локально"
      : "Модель " + model.name + " недоступна. Проверьте её локальную настройку.";
    invalidateResult();
  }

  function invalidateResult() {
    state.revision++;
    resultText.value = "";
    $("metricsText").textContent = "";
    if (state.uploadId) setStatus("Настройки изменены");
  }

  async function loadModels() {
    try {
      const response = await fetch("/api/models");
      if (!response.ok) throw new Error(await readError(response));
      const catalog = await response.json();
      state.models = catalog.models;
      modelSelect.replaceChildren(...state.models.map((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.name + (model.available ? "" : " — нет файлов");
        return option;
      }));
      modelSelect.value = catalog.default_model_id;
      maxTokensInput.max = catalog.max_new_tokens_limit;
      maxPixelsInput.max = catalog.max_pixels_limit;
      state.ready = true;
      applyDefaults();
    } catch (error) {
      state.modelsError = "Ошибка списка моделей: " + error.message;
      setStatus(state.modelsError);
    } finally { setBusy(false); }
  }

  function validateSettings(report = false) {
    // Пробельные символы совпадают с серверной проверкой Python str.strip().
    if (/^[\u0009-\u000D\u001C-\u0020\u0085\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]*$/u.test(promptInput.value)) {
      if (report) setStatus("Исправьте параметры модели: промпт должен содержать хотя бы один непробельный символ.");
      return false;
    }
    if (utf8.encode(promptInput.value).byteLength > MAX_PROMPT_BYTES) {
      if (report) setStatus("Исправьте параметры модели: размер промпта не должен превышать 8 КиБ в UTF-8.");
      return false;
    }
    const valid = report ? settingsForm.reportValidity() : settingsForm.checkValidity();
    if (!valid && report) setStatus("Исправьте параметры модели и повторите действие.");
    return valid;
  }

  async function refreshCachedResult() {
    if (!state.uploadId || state.busy || !validateSettings()) return;
    const revision = state.revision;
    try {
      const response = await fetch("/api/cached-result", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({upload_id: state.uploadId, ...selectedSettings()}),
      });
      if (!response.ok) throw new Error(await readError(response));
      const payload = await response.json();
      if (revision !== state.revision || state.busy) return;
      resultText.value = payload.text || "";
      setStatus(payload.cached ? "Текст выбранной модели загружен из кеша" : "Готово к распознаванию");
      if (payload.limit_reached) $("metricsText").textContent = "Сохранённый текст достиг лимита токенов. Увеличьте предел и повторите OCR.";
    } catch (error) {
      if (revision === state.revision) setStatus("Ошибка: " + error.message);
    }
  }

  function canLoadImage() {
    if (state.busy) return false;
    if (!state.ready) {
      setStatus(state.modelsError
        ? state.modelsError + ". Вставка недоступна до получения списка моделей."
        : "Дождитесь загрузки списка моделей и повторите вставку изображения.");
      return false;
    }
    return validateSettings(true);
  }

  async function handleFile(file, fromClipboard = false) {
    if (!file || !canLoadImage()) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setStatus("Изображение не загружено: размер файла не должен превышать 25 МиБ.");
      return;
    }
    if (fromClipboard) appendLog("Получено изображение из буфера обмена.");
    state.revision++;
    setBusy(true);
    setStatus("Загрузка изображения...");
    appendLog("Загрузка: " + file.name);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("settings", JSON.stringify(selectedSettings()));
      const response = await fetch("/api/upload", {method: "POST", body: form});
      if (!response.ok) throw new Error(await readError(response));
      const payload = await response.json();
      state.uploadId = payload.upload_id;
      imagePreview.src = payload.image_url;
      imagePreview.hidden = false;
      $("emptyState").hidden = true;
      resultText.value = payload.text || "";
      $("metricsText").textContent = payload.limit_reached ? "Сохранённый текст достиг лимита токенов." : "";
      setStatus(payload.cached ? "Текст выбранной модели загружен из кеша" : "Изображение загружено");
    } catch (error) {
      setStatus("Ошибка: " + error.message);
      appendLog("Ошибка загрузки: " + error.message);
    } finally { setBusy(false); }
  }

  async function recognize() {
    if (!state.uploadId || state.busy || !validateSettings(true)) return;
    state.revision++;
    setBusy(true);
    setStatus("Распознавание...");
    $("metricsText").textContent = "";
    try {
      const response = await fetch("/api/recognize", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({upload_id: state.uploadId, force: forceInput.checked, ...selectedSettings()}),
      });
      if (!response.ok) throw new Error(await readError(response));
      const payload = await response.json();
      state.jobId = payload.job_id;
      setBusy(true);
      listenJob(payload.job_id);
    } catch (error) {
      setStatus("Ошибка: " + error.message);
      appendLog(error.message);
      finishBusyState();
    }
  }

  function closeStream() {
    if (state.stream) state.stream.close();
    state.stream = null;
  }

  function listenJob(jobId) {
    closeStream();
    const stream = new EventSource("/api/stream/" + jobId);
    state.stream = stream;
    const isCurrent = () => state.stream === stream && state.jobId === jobId;
    stream.onopen = () => {
      if (isCurrent()) setStatus("Распознавание...");
    };
    stream.onmessage = (event) => {
      if (!isCurrent() || !event.data) return;
      const payload = JSON.parse(event.data);
      if (payload.type === "log") appendLog(payload.message);
      if (payload.type === "done") finishJob(payload);
    };
    stream.onerror = () => {
      if (!isCurrent()) return;
      setStatus(stream.readyState === EventSource.CLOSED
        ? "Соединение закрыто; состояние задачи неизвестно. Обновите страницу для восстановления интерфейса."
        : "Связь прервана; ожидаем восстановления и завершения распознавания.");
      appendLog("Поток логов оборвался; сервер мог продолжить распознавание.");
    };
  }

  function finishJob(payload) {
    if (payload.status === "ok") {
      const result = payload.result;
      resultText.value = result.text || "";
      const metrics = result.metrics || {};
      const elapsed = Number(payload.elapsed_seconds).toFixed(2);
      setStatus(result.cached ? "Текст выбранной модели загружен из кеша" : "Распознавание завершено за " + elapsed + " с");
      if (!result.cached) {
        const seconds = (value) => Number(value || 0).toFixed(2);
        $("metricsText").textContent =
          "Загрузка: " + seconds(metrics.load_seconds) + " с · Подготовка: " + seconds(metrics.prepare_seconds) +
          " с · Генерация: " + seconds(metrics.generate_seconds) + " с · " +
          Number(metrics.tokens_per_second || 0).toFixed(1) + " токен/с · " + result.device;
      }
      if (metrics.limit_reached) {
        $("metricsText").textContent += " · Достигнут лимит токенов: текст может быть обрезан.";
      }
      appendLog("Завершено: " + result.model_id + ", " + elapsed + " с.");
    } else {
      const message = errorMessage(payload, "error");
      setStatus("Ошибка: " + message);
      appendLog("Ошибка OCR: " + message);
    }
    finishBusyState();
  }

  function finishBusyState() {
    closeStream();
    state.jobId = null;
    setBusy(false);
  }

  dropZone.addEventListener("click", () => {
    if (!state.busy && state.ready) fileInput.click();
  });
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (!state.busy) dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault(); dropZone.classList.remove("drag-over"); handleFile(event.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => { handleFile(fileInput.files[0]); fileInput.value = ""; });
  document.addEventListener("paste", (event) => {
    const file = getClipboardImage(event);
    if (!file) return;
    event.preventDefault();
    handleFile(file, true);
  });
  settingsForm.addEventListener("submit", (event) => event.preventDefault());
  modelSelect.addEventListener("change", () => { applyDefaults(); refreshCachedResult(); });
  $("resetSettingsBtn").addEventListener("click", () => { applyDefaults(); refreshCachedResult(); });
  for (const input of [promptInput, maxTokensInput, maxPixelsInput, quantizationInput]) {
    input.addEventListener("input", invalidateResult);
    input.addEventListener("change", refreshCachedResult);
  }
  recognizeBtn.addEventListener("click", recognize);
  $("clearLogsBtn").addEventListener("click", () => { logConsole.textContent = ""; });
  setBusy(false);
  loadModels();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeApp, {once: true});
} else {
  initializeApp();
}
