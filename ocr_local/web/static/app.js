"use strict";

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const emptyState = document.getElementById("emptyState");
const imagePreview = document.getElementById("imagePreview");
const resultText = document.getElementById("resultText");
const recognizeBtn = document.getElementById("recognizeBtn");
const statusText = document.getElementById("statusText");
const logConsole = document.getElementById("logConsole");
const clearLogsBtn = document.getElementById("clearLogsBtn");

const state = {
  imagePath: null,
  imageUrl: null,
  jobId: null,
  stream: null,
};

function setStatus(text) {
  statusText.textContent = text;
}

function appendLog(text) {
  if (!text) {
    return;
  }
  logConsole.textContent += text.endsWith("\n") ? text : `${text}\n`;
  logConsole.scrollTop = logConsole.scrollHeight;
}

function setBusy(isBusy) {
  recognizeBtn.disabled = isBusy || !state.imagePath;
  recognizeBtn.textContent = isBusy ? "Распознается..." : "Распознать";
}

function closeStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  setBusy(true);
  setStatus("Загрузка изображения...");
  appendLog(`Загрузка: ${file.name}`);
  const response = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await readError(response);
    throw new Error(error);
  }
  return response.json();
}

async function recognize() {
  if (!state.imagePath || state.jobId) {
    return;
  }
  setBusy(true);
  setStatus("Распознавание...");
  appendLog("Запущено распознавание.");
  const response = await fetch("/api/recognize", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({image_path: state.imagePath, force: false}),
  });
  if (!response.ok) {
    const error = await readError(response);
    setStatus(`Ошибка: ${error}`);
    setBusy(false);
    return;
  }
  const payload = await response.json();
  state.jobId = payload.job_id;
  listenJob(payload.job_id);
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.detail || response.statusText;
  } catch (err) {
    return response.statusText;
  }
}

function applyUpload(payload) {
  closeStream();
  state.jobId = null;
  state.imagePath = payload.image_path;
  state.imageUrl = payload.image_url;
  imagePreview.src = payload.image_url;
  imagePreview.hidden = false;
  emptyState.hidden = true;
  resultText.value = payload.text || "";
  setStatus(payload.cached ? "Текст загружен из кеша" : "Изображение загружено");
  if (payload.cached) {
    appendLog("OCR найден в кеше.");
  }
  setBusy(false);
}

function listenJob(jobId) {
  closeStream();
  const stream = new EventSource(`/api/stream/${jobId}`);
  state.stream = stream;

  stream.onmessage = (event) => {
    if (!event.data) {
      return;
    }
    const payload = JSON.parse(event.data);
    if (payload.type === "log") {
      appendLog(payload.message);
      return;
    }
    if (payload.type === "done") {
      finishJob(payload);
    }
  };

  stream.onerror = () => {
    appendLog("Поток логов оборвался.");
    setStatus("Ошибка: поток логов оборвался");
    finishBusyState();
  };
}

function finishJob(payload) {
  if (payload.status === "ok") {
    const result = payload.result || {};
    resultText.value = result.text || "";
    setStatus(result.cached ? "Текст загружен из кеша" : "Распознавание завершено");
    appendLog("Распознавание завершено.");
  } else {
    const error = payload.error || "неизвестная ошибка";
    setStatus(`Ошибка: ${error}`);
    appendLog(`Ошибка OCR: ${error}`);
  }
  finishBusyState();
}

function finishBusyState() {
  closeStream();
  state.jobId = null;
  setBusy(false);
}

async function handleFile(file) {
  if (!file) {
    return;
  }
  try {
    const payload = await uploadFile(file);
    applyUpload(payload);
  } catch (err) {
    setStatus(`Ошибка: ${err.message}`);
    appendLog(`Ошибка загрузки: ${err.message}`);
    setBusy(false);
  }
}

function getClipboardImage(event) {
  const items = Array.from(event.clipboardData?.items || []);
  for (const item of items) {
    if (item.kind === "file" && item.type.startsWith("image/")) {
      const file = item.getAsFile();
      const extension = clipboardExtension(file?.type);
      if (!file || !extension) {
        return null;
      }
      return new File([file], `clipboard-${Date.now()}.${extension}`, {
        type: file.type || "image/png",
      });
    }
  }
  return null;
}

function clipboardExtension(mimeType) {
  const extensions = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/bmp": "bmp",
  };
  return extensions[mimeType] || null;
}

dropZone.addEventListener("click", () => {
  fileInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = event.dataTransfer.files[0];
  handleFile(file);
});

fileInput.addEventListener("change", () => {
  handleFile(fileInput.files[0]);
});

document.addEventListener("paste", (event) => {
  const imageFile = getClipboardImage(event);
  if (!imageFile) {
    return;
  }
  event.preventDefault();
  appendLog("Получено изображение из буфера обмена.");
  handleFile(imageFile);
});

recognizeBtn.addEventListener("click", () => {
  recognize();
});

clearLogsBtn.addEventListener("click", () => {
  logConsole.textContent = "";
});
