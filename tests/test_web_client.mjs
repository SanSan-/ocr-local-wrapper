import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {setImmediate} from "node:timers/promises";
import test from "node:test";
import vm from "node:vm";

const sourcePath = process.env.OCR_CLIENT_SOURCE || new URL("../ocr_local/web/static/app.js", import.meta.url);
const source = await readFile(sourcePath, "utf8");
const htmlPath = process.env.OCR_CLIENT_HTML || new URL("../ocr_local/web/static/index.html", import.meta.url);
const html = await readFile(htmlPath, "utf8");
const catalog = {
  default_model_id: "paddleocr-vl-1.6", max_new_tokens_limit: 32768, max_pixels_limit: 16777216,
  models: [
    {id: "glm-ocr", name: "GLM-OCR", prompt: "Text Recognition:", max_new_tokens: 8192, max_pixels: 2007040, min_pixels: 12544},
    {id: "paddleocr-vl-1.6", name: "PaddleOCR-VL-1.6", prompt: "OCR:", max_new_tokens: 2048, max_pixels: 1003520, min_pixels: 112896},
  ],
};
const json = (value, status = 200) => Response.json(value, {status});
const settle = async () => { for (let i = 0; i < 4; i++) await setImmediate(); };
const deferred = () => { let resolve; const promise = new Promise(r => {resolve = r;}); return {promise, resolve}; };
const imageItem = (type = "image/png", text = "image") => ({
  kind: "file", type, getAsFile: () => new File([text], "image", {type}),
});

function markupElement(id) {
  const match = html.match(new RegExp(`<([a-z][\\w-]*)\\b([^>]*\\bid="${id}"[^>]*)>`, "i"));
  assert.ok(match, "Элемент отсутствует в HTML: " + id);
  const attributes = new Map(Array.from(match[2].matchAll(/([:\w-]+)(?:="([^"]*)")?/g),
    item => [item[1], item[2] ?? ""]));
  return {tagName: match[1].toUpperCase(), attributes};
}

function element({tagName = "", attributes = new Map()} = {}) {
  const listeners = new Map();
  return {
    tagName, value: "", textContent: "", hidden: attributes.has("hidden"),
    disabled: attributes.has("disabled"), checked: false, valid: true, files: [], clicks: 0,
    addEventListener(name, handler, options = {}) { listeners.set(name, {handler, once: options.once}); },
    emit(name, fields = {}) {
      const event = {target: this, defaultPrevented: false, preventDefault() {this.defaultPrevented = true;}, ...fields};
      const listener = listeners.get(name);
      if (listener?.once) listeners.delete(name);
      listener?.handler(event);
      return event;
    },
    setAttribute(name, value) {this[name] = value;},
    replaceChildren(...children) {this.children = children;},
    reportValidity() {this.reports = (this.reports || 0) + 1; return this.valid;},
    checkValidity() {return this.valid;},
    set innerHTML(value) {throw new Error("Разметка не должна исполняться: " + value);},
    insertAdjacentHTML() {throw new Error("Разметка не должна исполняться");},
    click() {if (!this.disabled) {this.clicks++; this.emit("click");}},
    pressKey(key) {
      const down = this.emit("keydown", {key});
      if (this.tagName === "BUTTON" && key === "Enter" && !down.defaultPrevented) this.click();
      const up = this.emit("keyup", {key});
      if (this.tagName === "BUTTON" && key === " " && !down.defaultPrevented && !up.defaultPrevented) this.click();
    },
    classList: {add() {}, remove() {}},
  };
}

async function client({models = () => json(catalog), upload, recognize, cached,
  readyState = "loading", autoReady = true} = {}) {
  const elements = new Map();
  const document = element();
  document.readyState = readyState;
  const el = id => {
    if (!elements.has(id)) elements.set(id, element(markupElement(id)));
    return elements.get(id);
  };
  document.getElementById = id => {
    assert.notEqual(document.readyState, "loading", "Нельзя читать элементы до готовности DOM");
    return el(id);
  };
  document.createElement = element;
  el("modelSettings").elements = ["modelSelect", "promptInput", "maxTokensInput", "maxPixelsInput",
    "quantizationInput", "forceInput", "resetSettingsBtn"].map(el);
  const calls = [];
  const streams = [];
  const context = vm.createContext({
    document, File, FormData, TextEncoder,
    EventSource: class {
      static CLOSED = 2;
      constructor(url, options) {this.url = url; this.options = options; streams.push(this);}
      close() {this.closed = true;}
    },
    fetch: async (url, options = {}) => {
      calls.push({url, ...options});
      if (url === "/api/models") return models();
      if (url === "/api/upload") {
        return upload ? upload(options) : json({
          upload_id: "image-id.png", image_url: "/api/image/image.png",
          text: "", cached: false,
        });
      }
      if (url === "/api/recognize") return recognize ? recognize(options) : json({job_id: "test-job"});
      if (url === "/api/cached-result") return cached ? cached(options) : json({cached: false, text: ""});
      throw new Error("Неожиданный запрос: " + url);
    },
  });
  vm.runInContext(source, context, {filename: "app.js"});
  const domReady = () => { document.readyState = "interactive"; document.emit("DOMContentLoaded"); };
  if (autoReady && readyState === "loading") domReady();
  await settle();
  return {
    el, calls, streams, domReady,
    uploads: () => calls.filter(call => call.url === "/api/upload"),
    paste: items => document.emit("paste", {clipboardData: {items}}),
  };
}

test("Разметка использует доступную кнопку и отдельный скрытый выбор файла", () => {
  const button = markupElement("dropZone");
  const input = markupElement("fileInput");
  assert.equal(button.tagName, "BUTTON");
  assert.equal(button.attributes.get("type"), "button");
  assert.ok(button.attributes.get("aria-label")?.trim());
  assert.equal(button.attributes.has("tabindex"), false);
  assert.equal(input.tagName, "INPUT");
  assert.equal(input.attributes.get("type"), "file");
  assert.ok(input.attributes.has("hidden"));
  assert.ok(input.attributes.get("aria-label")?.trim());
  const content = html.match(/<button\b[^>]*\bid="dropZone"[^>]*>([\s\S]*?)<\/button>/i)[1];
  assert.doesNotMatch(content, /<(?:input|select|textarea|button|div)\b/i);
});

test("Поле результата имеет доступное имя в фактической разметке", () => {
  const result = markupElement("resultText");
  assert.equal(result.tagName, "TEXTAREA");
  assert.equal(result.attributes.get("aria-label"), "Результат распознавания");
});

test("Инициализация ждёт DOM и получает каталог ровно один раз", async () => {
  const c = await client({autoReady: false});
  assert.equal(c.calls.length, 0);
  assert.equal(c.el("dropZone").disabled, true);
  c.domReady();
  await settle();
  assert.equal(c.calls.filter(call => call.url === "/api/models").length, 1);
  assert.equal(c.el("modelSelect").value, catalog.default_model_id);
  assert.equal(c.el("dropZone").disabled, false);
  c.domReady();
  await settle();
  assert.equal(c.calls.length, 1);
});

test("Скрипт инициализирует уже готовый документ без ожидания нового события", async () => {
  const c = await client({readyState: "complete"});
  assert.equal(c.calls.filter(call => call.url === "/api/models").length, 1);
  assert.equal(c.el("dropZone").disabled, false);
  c.domReady();
  await settle();
  assert.equal(c.calls.length, 1);
});

for (const activation of ["click", "Enter", " "]) {
  test("Кнопка открывает один выбор файла при активации " + JSON.stringify(activation), async () => {
    const c = await client();
    const button = c.el("dropZone");
    if (activation === "click") button.click();
    else button.pressKey(activation);
    assert.equal(c.el("fileInput").clicks, 1);
    assert.equal(c.uploads().length, 0);
    assert.equal(c.calls.some(call => call.url === "/api/recognize"), false);
  });
}

test("Выбор файла заблокирован до каталога, при загрузке, OCR и восстановлении SSE", async () => {
  const models = deferred();
  const upload = deferred();
  const c = await client({models: () => models.promise, upload: () => upload.promise});
  const assertBlocked = () => {
    assert.equal(c.el("dropZone").disabled, true);
    c.el("dropZone").click();
    c.el("dropZone").pressKey("Enter");
    c.el("dropZone").pressKey(" ");
    assert.equal(c.el("fileInput").clicks, 0);
  };
  assertBlocked();
  models.resolve(json(catalog));
  await settle();
  assert.equal(c.el("dropZone").disabled, false);
  c.paste([imageItem()]);
  assertBlocked();
  upload.resolve(json({upload_id: "button-id.png", image_url: "/button.png", text: ""}));
  await settle();
  c.el("recognizeBtn").click();
  await settle();
  assertBlocked();
  c.streams[0].onerror();
  assertBlocked();
  c.streams[0].onopen();
  assertBlocked();
  c.streams[0].onmessage({data: JSON.stringify({type: "done", status: "error", error: "Отказ OCR"})});
  assert.equal(c.el("dropZone").disabled, false);
  c.el("dropZone").pressKey("Enter");
  assert.equal(c.el("fileInput").clicks, 1);
  assert.equal(c.uploads().length, 1);
  assert.equal(c.calls.filter(call => call.url === "/api/recognize").length, 1);
});

for (const [mime, extension] of [["image/png", "png"], ["image/jpeg", "jpg"], ["image/webp", "webp"], ["image/bmp", "bmp"]]) {
  test("Вставка " + mime + " передаёт один файл с параметрами и не запускает OCR", async () => {
    const c = await client();
    c.el("promptInput").value = "Ручной промпт";
    c.el("maxTokensInput").value = "128";
    c.el("maxPixelsInput").value = "200000";
    const event = c.paste([imageItem(mime)]);
    await settle();
    assert.equal(event.defaultPrevented, true);
    assert.equal(c.uploads().length, 1);
    const body = c.uploads()[0].body;
    assert.equal(body.get("file").type, mime);
    assert.ok(body.get("file").name.endsWith("." + extension));
    assert.deepEqual(JSON.parse(body.get("settings")), {
      model_id: "paddleocr-vl-1.6", prompt: "Ручной промпт", max_new_tokens: 128,
      max_pixels: 200000, quantization_enabled: false,
    });
    assert.equal(c.el("imagePreview").hidden, false);
    assert.equal(c.el("promptInput").value, "Ручной промпт");
    assert.equal(c.el("recognizeBtn").disabled, false);
    assert.equal(c.calls.some(call => call.url === "/api/recognize"), false);
    assert.match(c.el("logConsole").textContent, /Получено изображение из буфера/);
  });
}

test("Поиск пропускает текст, неподдержанный и недоступный элементы", async () => {
  const c = await client();
  c.paste([{kind: "string", type: "text/plain"}, imageItem("image/gif"),
    {kind: "file", type: "image/png", getAsFile: () => null},
    imageItem("image/png", "first"), imageItem("image/jpeg", "second")]);
  await settle();
  assert.equal(c.uploads().length, 1);
  assert.equal(await c.uploads()[0].body.get("file").text(), "first");
});

test("Текстовый, пустой и неподдержанный буфер не отменяются и не сбрасывают результат", async () => {
  const c = await client();
  c.el("resultText").value = "Сохранённый результат";
  c.el("imagePreview").src = "/previous.png";
  for (const items of [[], [{kind: "string", type: "text/plain"}], [imageItem("image/gif")]]) {
    assert.equal(c.paste(items).defaultPrevented, false);
  }
  await settle();
  assert.equal(c.uploads().length, 0);
  assert.equal(c.el("resultText").value, "Сохранённый результат");
  assert.equal(c.el("imagePreview").src, "/previous.png");
});

test("До готовности каталога есть объяснение; после готовности повторная вставка работает", async () => {
  const pending = deferred();
  const c = await client({models: () => pending.promise});
  c.paste([imageItem()]);
  assert.equal(c.uploads().length, 0);
  assert.match(c.el("statusText").textContent, /Дождитесь загрузки списка моделей/);
  pending.resolve(json(catalog));
  await settle();
  c.paste([imageItem()]);
  await settle();
  assert.equal(c.uploads().length, 1);
});

test("Отказ каталога сохраняется в объяснении недоступной вставки", async () => {
  const c = await client({models: () => json({detail: "Каталог недоступен"}, 503)});
  assert.equal(c.el("dropZone").disabled, true);
  c.el("dropZone").click();
  c.el("dropZone").pressKey("Enter");
  assert.equal(c.el("fileInput").clicks, 0);
  c.paste([imageItem()]);
  assert.match(c.el("statusText").textContent, /Каталог недоступен/);
  assert.match(c.el("statusText").textContent, /Вставка недоступна/);
  assert.equal(c.uploads().length, 0);
});

test("Неверные параметры сохраняются, с объяснением отказа и без upload", async () => {
  const c = await client();
  c.el("modelSettings").valid = false;
  c.el("promptInput").value = "";
  c.el("maxTokensInput").value = "0";
  c.el("imagePreview").src = "/previous.png";
  c.paste([imageItem()]);
  assert.match(c.el("statusText").textContent, /Исправьте параметры модели/);
  assert.equal(c.el("promptInput").value, "");
  assert.equal(c.el("maxTokensInput").value, "0");
  assert.equal(c.el("imagePreview").src, "/previous.png");
  assert.equal(c.uploads().length, 0);
});

test("Повторная вставка во время upload не создаёт второй запрос", async () => {
  const pending = deferred();
  const c = await client({upload: () => pending.promise});
  c.paste([imageItem()]);
  c.paste([imageItem("image/jpeg")]);
  assert.equal(c.uploads().length, 1);
  assert.equal(c.el("statusText").textContent, "Загрузка изображения...");
  pending.resolve(json({upload_id: "first-id.png", image_url: "/first.png", text: ""}));
  await settle();
  assert.equal(c.el("imagePreview").src, "/first.png");
});

test("Вставка во время OCR не меняет изображение или активную задачу", async () => {
  const c = await client();
  c.paste([imageItem()]);
  await settle();
  const preview = c.el("imagePreview").src;
  c.el("recognizeBtn").click();
  await settle();
  c.paste([imageItem("image/jpeg")]);
  await settle();
  assert.equal(c.uploads().length, 1);
  assert.equal(c.calls.filter(call => call.url === "/api/recognize").length, 1);
  assert.equal(c.el("imagePreview").src, preview);
  assert.equal(c.el("recognizeBtn").disabled, true);
  assert.equal(c.streams.length, 1);
});

test("Смена модели и повторная вставка используют только кеш новых настроек", async () => {
  const c = await client({upload: options => {
    const settings = JSON.parse(options.body.get("settings"));
    return json({upload_id: "image-id.png", image_url: "/image.png", cached: true, text: settings.model_id});
  }});
  c.paste([imageItem()]);
  await settle();
  assert.equal(c.el("resultText").value, "paddleocr-vl-1.6");
  c.el("modelSelect").value = "glm-ocr";
  c.el("modelSelect").emit("change");
  await settle();
  c.paste([imageItem()]);
  await settle();
  assert.equal(c.el("resultText").value, "glm-ocr");
  assert.equal(c.el("promptInput").value, "Text Recognition:");
  assert.equal(c.uploads().length, 2);
});

test("Выбор файла и перетаскивание продолжают использовать общий upload", async () => {
  const c = await client();
  c.el("fileInput").files = [new File(["choose"], "choose.png", {type: "image/png"})];
  c.el("fileInput").emit("change");
  await settle();
  c.el("dropZone").emit("drop", {dataTransfer: {files: [new File(["drop"], "drop.png", {type: "image/png"})]}});
  await settle();
  assert.equal(c.uploads().length, 2);
  assert.equal(await c.uploads()[0].body.get("file").text(), "choose");
  assert.equal(await c.uploads()[1].body.get("file").text(), "drop");
});

test("Ошибка upload сохраняет прежнее изображение и позволяет повторить вставку", async () => {
  let fail = true;
  const c = await client({upload: () => {
    if (fail) return json({detail: "Тестовый отказ"}, 400);
    return json({upload_id: "retry-id.png", image_url: "/retry.png", text: ""});
  }});
  c.el("imagePreview").src = "/previous.png";
  c.paste([imageItem()]);
  await settle();
  assert.equal(c.el("imagePreview").src, "/previous.png");
  assert.match(c.el("statusText").textContent, /Тестовый отказ/);
  fail = false;
  c.paste([imageItem()]);
  await settle();
  assert.equal(c.uploads().length, 2);
  assert.equal(c.el("imagePreview").src, "/retry.png");
});

for (const prompt of ["   ", "\t\n", "\u00a0\u2003", "\u0085", "\u001c\u001d\u001e\u001f"]) {
  test("Пробельный промпт блокирует все запросы и исправление восстанавливает работу: " + JSON.stringify(prompt), async () => {
    const c = await client();
    c.paste([imageItem()]);
    await settle();
    c.el("promptInput").value = prompt;
    c.el("resultText").value = "Прежний результат";
    const preview = c.el("imagePreview").src;
    const requests = c.calls.length;
    c.el("promptInput").emit("change");
    c.el("recognizeBtn").click();
    c.paste([imageItem()]);
    c.el("fileInput").files = [new File(["choose"], "choose.png", {type: "image/png"})];
    c.el("fileInput").emit("change");
    c.el("dropZone").emit("drop", {dataTransfer: {files: [new File(["drop"], "drop.png", {type: "image/png"})]}});
    await settle();
    assert.equal(c.calls.length, requests);
    assert.equal(c.el("promptInput").value, prompt);
    assert.equal(c.el("resultText").value, "Прежний результат");
    assert.equal(c.el("imagePreview").src, preview);
    assert.match(c.el("statusText").textContent, /промпт/i);
    c.el("promptInput").value = "  Ручной промпт  ";
    c.el("promptInput").emit("input");
    c.el("promptInput").emit("change");
    await settle();
    c.paste([imageItem()]);
    await settle();
    assert.equal(c.uploads().length, 2);
    assert.equal(JSON.parse(c.uploads()[1].body.get("settings")).prompt, "  Ручной промпт  ");
    assert.equal(c.calls.filter(call => call.url === "/api/cached-result").length, 1);
    c.el("recognizeBtn").click();
    await settle();
    assert.equal(c.calls.filter(call => call.url === "/api/recognize").length, 1);
  });
}

for (const status of ["ok", "error"]) {
  test("Разрыв SSE сохраняет блокировку до done/" + status + " текущей задачи", async () => {
    const c = await client();
    c.paste([imageItem()]);
    await settle();
    c.el("recognizeBtn").click();
    await settle();
    const stream = c.streams[0];
    const preview = c.el("imagePreview").src;
    stream.readyState = 0;
    stream.onerror();
    assert.notEqual(stream.closed, true);
    assert.equal(c.el("recognizeBtn").disabled, true);
    assert.equal(c.el("fileInput").disabled, true);
    assert.ok(c.el("modelSettings").elements.every(control => control.disabled));
    c.paste([imageItem()]);
    c.el("dropZone").emit("drop", {dataTransfer: {files: [new File(["next"], "next.png", {type: "image/png"})]}});
    c.el("recognizeBtn").click();
    await settle();
    assert.equal(c.uploads().length, 1);
    assert.equal(c.calls.filter(call => call.url === "/api/recognize").length, 1);
    assert.equal(c.el("imagePreview").src, preview);
    stream.onopen();
    const done = status === "ok"
      ? {type: "done", status, result: {text: "Готово", model_id: "paddleocr-vl-1.6", metrics: {}}, elapsed_seconds: 1}
      : {type: "done", status, error: "Проверочный отказ"};
    stream.onmessage({data: JSON.stringify(done)});
    assert.equal(c.el("recognizeBtn").disabled, false);
    assert.equal(c.el("fileInput").disabled, false);
    assert.ok(c.el("modelSettings").elements.every(control => !control.disabled));
    assert.equal(stream.closed, true);
    assert.match(c.el("statusText").textContent, status === "ok" ? /завершено/ : /Проверочный отказ/);
    c.el("recognizeBtn").click();
    await settle();
    const currentStatus = c.el("statusText").textContent;
    stream.onerror();
    stream.onopen();
    stream.onmessage({data: JSON.stringify(done)});
    assert.equal(c.el("recognizeBtn").disabled, true);
    assert.equal(c.el("fileInput").disabled, true);
    assert.equal(c.el("statusText").textContent, currentStatus);
    assert.notEqual(c.streams[1].closed, true);
  });
}

test("Закрытый SSE сообщает неизвестное состояние и не снимает блокировку", async () => {
  const c = await client();
  c.paste([imageItem()]);
  await settle();
  c.el("recognizeBtn").click();
  await settle();
  c.streams[0].readyState = 2;
  c.streams[0].onerror();
  assert.equal(c.el("recognizeBtn").disabled, true);
  assert.equal(c.el("fileInput").disabled, true);
  assert.match(c.el("statusText").textContent, /состояние задачи неизвестно/i);
  assert.match(c.el("statusText").textContent, /обновите страницу/i);
});

test("Нативная проверка пределов сохраняется при корректном промпте", async () => {
  const c = await client();
  c.paste([imageItem()]);
  await settle();
  c.el("promptInput").value = "OCR:";
  c.el("modelSettings").valid = false;
  c.el("maxTokensInput").value = "0";
  const requests = c.calls.length;
  c.el("maxTokensInput").emit("change");
  c.el("recognizeBtn").click();
  c.paste([imageItem()]);
  await settle();
  assert.equal(c.calls.length, requests);
  assert.match(c.el("statusText").textContent, /Исправьте параметры модели/);
  assert.equal(c.el("maxTokensInput").value, "0");
});

test("U+FEFF сохраняется как непробельный символ по серверному контракту", async () => {
  const c = await client();
  c.el("promptInput").value = "\ufeff";
  c.paste([imageItem()]);
  await settle();
  assert.equal(c.uploads().length, 1);
  assert.equal(JSON.parse(c.uploads()[0].body.get("settings")).prompt, "\ufeff");
});

test("Распознавание и кеш используют upload_id; штатный EventSource не передаёт секрет в URL", async () => {
  const c = await client();
  c.paste([imageItem()]);
  await settle();
  c.el("promptInput").emit("change");
  await settle();
  c.el("recognizeBtn").click();
  await settle();
  for (const call of c.calls.filter(call => ["/api/recognize", "/api/cached-result"].includes(call.url))) {
    const body = JSON.parse(call.body);
    assert.equal(body.upload_id, "image-id.png");
    assert.equal("image_path" in body, false);
    assert.equal("model_path" in body, false);
    assert.equal("output_path" in body, false);
  }
  assert.equal(c.streams.length, 1);
  assert.equal(c.streams[0].url, "/api/stream/test-job");
  assert.equal(c.streams[0].options, undefined);
  assert.equal(c.calls.some(call => call.url.includes("?")), false);
});

test("Недоступная модель объясняется без чтения или показа серверного пути", async () => {
  const publicCatalog = structuredClone(catalog);
  for (const model of publicCatalog.models) {
    model.available = false;
    Object.defineProperty(model, "path", {get() {throw new Error("Внутренний путь недоступен клиенту");}});
  }
  const c = await client({models: () => ({ok: true, json: async () => publicCatalog})});
  assert.match(c.el("modelHint").textContent, /PaddleOCR-VL-1\.6 недоступна/);
  assert.match(c.el("modelHint").textContent, /локальную настройку/);
  assert.equal(c.el("modelHint").title, undefined);
  c.el("modelSelect").value = "glm-ocr";
  c.el("modelSelect").emit("change");
  assert.match(c.el("modelHint").textContent, /GLM-OCR недоступна/);
});

for (const [character, bytes] of [["a", 1], ["я", 2], ["😀", 4]]) {
  test("Предел промпта считается в UTF-8, включая границу 8 КиБ: " + character, async () => {
    const c = await client();
    const boundary = character.repeat(8192 / bytes);
    c.el("promptInput").value = boundary;
    c.paste([imageItem()]);
    await settle();
    assert.equal(c.uploads().length, 1);
    assert.equal(JSON.parse(c.uploads()[0].body.get("settings")).prompt, boundary);
    c.el("promptInput").value = boundary + "a";
    c.el("resultText").value = "Прежний результат";
    const preview = c.el("imagePreview").src;
    const requests = c.calls.length;
    c.el("promptInput").emit("change");
    c.el("recognizeBtn").click();
    c.paste([imageItem()]);
    await settle();
    assert.equal(c.calls.length, requests);
    assert.match(c.el("statusText").textContent, /8 КиБ.*UTF-8/);
    assert.equal(c.el("promptInput").value, boundary + "a");
    assert.equal(c.el("resultText").value, "Прежний результат");
    assert.equal(c.el("imagePreview").src, preview);
    c.el("promptInput").value = "  Допустимый промпт  ";
    c.paste([imageItem()]);
    await settle();
    assert.equal(c.uploads().length, 2);
    assert.equal(JSON.parse(c.uploads()[1].body.get("settings")).prompt, "  Допустимый промпт  ");
  });
}

test("Лимит файла 25 МиБ действует для вставки, выбора и перетаскивания до запроса", async () => {
  const c = await client();
  c.paste([imageItem()]);
  await settle();
  const preview = c.el("imagePreview").src;
  c.el("resultText").value = "Прежний текст";
  c.el("promptInput").value = "Ручной промпт";
  const oversized = new File([new Uint8Array(25 * 1024 * 1024 + 1)], "large.png", {type: "image/png"});
  c.paste([{kind: "file", type: "image/png", getAsFile: () => oversized}]);
  c.el("fileInput").files = [oversized];
  c.el("fileInput").emit("change");
  c.el("dropZone").emit("drop", {dataTransfer: {files: [oversized]}});
  await settle();
  assert.equal(c.uploads().length, 1);
  assert.match(c.el("statusText").textContent, /25 МиБ/);
  assert.equal(c.el("imagePreview").src, preview);
  assert.equal(c.el("resultText").value, "Прежний текст");
  assert.equal(c.el("promptInput").value, "Ручной промпт");
  assert.equal(c.el("recognizeBtn").disabled, false);
  c.el("fileInput").files = [new File([oversized.slice(0, 25 * 1024 * 1024)], "boundary.png", {type: "image/png"})];
  c.el("fileInput").emit("change");
  await settle();
  assert.equal(c.uploads().length, 2);
  assert.equal(c.uploads()[1].body.get("file").size, 25 * 1024 * 1024);
});

for (const [status, code, detail] of [
  [413, "upload_too_large", "Изображение превышает допустимый размер."],
  [429, "upload_busy", "Другая загрузка ещё выполняется."],
  [507, "storage_quota_exceeded", "Недостаточно места. Освободите место вручную."],
]) {
  test("Отказ upload " + status + " сохраняет изображение, результат, настройки и прежний upload_id", async () => {
    let uploads = 0;
    const c = await client({upload: () => {
      if (++uploads === 1) return json({upload_id: "preserved-id.png", image_url: "/api/image/preserved-id.png", text: "Прежний текст", cached: true});
      return Response.json({detail, code, diagnostic_id: "diag-upload"}, {status, headers: {"Retry-After": "12"}});
    }});
    c.paste([imageItem()]);
    await settle();
    c.el("promptInput").value = "Ручной промпт";
    c.el("maxTokensInput").value = "128";
    c.el("maxPixelsInput").value = "200000";
    c.el("quantizationInput").checked = true;
    c.el("forceInput").checked = true;
    c.el("metricsText").textContent = "Прежние метрики";
    const preview = c.el("imagePreview").src;
    c.paste([imageItem("image/jpeg")]);
    await settle();
    assert.equal(c.el("imagePreview").src, preview);
    assert.equal(c.el("resultText").value, "Прежний текст");
    assert.equal(c.el("metricsText").textContent, "Прежние метрики");
    assert.equal(c.el("promptInput").value, "Ручной промпт");
    assert.equal(c.el("maxTokensInput").value, "128");
    assert.equal(c.el("maxPixelsInput").value, "200000");
    assert.equal(c.el("quantizationInput").checked, true);
    assert.equal(c.el("forceInput").checked, true);
    assert.equal(c.el("recognizeBtn").disabled, false);
    assert.ok(c.el("statusText").textContent.includes(detail));
    assert.ok(c.el("statusText").textContent.includes(code));
    assert.match(c.el("logConsole").textContent, /diag-upload/);
    if (status === 429) assert.match(c.el("statusText").textContent, /через 12 с/);
    c.el("recognizeBtn").click();
    await settle();
    const body = JSON.parse(c.calls.find(call => call.url === "/api/recognize").body);
    assert.equal(body.upload_id, "preserved-id.png");
    assert.equal(body.prompt, "Ручной промпт");
    assert.equal(body.force, true);
  });
}

test("HTTP и SSE показывают безопасную диагностику как текст, результат не исполняет разметку", async () => {
  const markup = '<img src=x onerror="alert(1)">';
  const c = await client();
  c.paste([imageItem()]);
  await settle();
  c.el("recognizeBtn").click();
  await settle();
  c.streams[0].onmessage({data: JSON.stringify({type: "log", message: markup})});
  c.streams[0].onmessage({data: JSON.stringify({type: "done", status: "ok", result: {
    text: markup, model_id: "paddleocr-vl-1.6", metrics: {},
  }, elapsed_seconds: 1})});
  assert.equal(c.el("resultText").value, markup);
  assert.ok(c.el("logConsole").textContent.includes(markup));
  c.el("recognizeBtn").click();
  await settle();
  c.streams[1].onmessage({data: JSON.stringify({type: "done", status: "error", error: markup,
    code: "result_too_large", diagnostic_id: "diag-sse"})});
  assert.equal(c.el("resultText").value, markup);
  assert.ok(c.el("statusText").textContent.includes(markup));
  assert.match(c.el("logConsole").textContent, /result_too_large.*diag-sse/);
  assert.equal(c.el("recognizeBtn").disabled, false);
  const failed = await client({upload: () => json({detail: markup, code: "invalid_image", diagnostic_id: "diag-http"}, 400)});
  failed.paste([imageItem()]);
  await settle();
  assert.ok(failed.el("statusText").textContent.includes(markup));
  assert.match(failed.el("logConsole").textContent, /invalid_image.*diag-http/);
});

test("Ответ без публичной ошибки не отражает техническое тело или statusText", async () => {
  const c = await client({upload: () => new Response("Traceback C:\\private\\file.py", {
    status: 500, statusText: "Internal path C:\\private\\file.py",
  })});
  c.paste([imageItem()]);
  await settle();
  assert.match(c.el("statusText").textContent, /Не удалось выполнить запрос/);
  assert.doesNotMatch(c.el("statusText").textContent, /Traceback|private/);
  assert.doesNotMatch(c.el("logConsole").textContent, /Traceback|private/);
});

test("Отправка формы отменяется внешним обработчиком без перезагрузки и запросов", async () => {
  const c = await client();
  const requests = c.calls.length;
  assert.equal(c.el("modelSettings").emit("submit").defaultPrevented, true);
  await settle();
  assert.equal(c.calls.length, requests);
});
