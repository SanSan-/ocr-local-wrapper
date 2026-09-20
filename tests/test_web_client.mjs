import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {setImmediate} from "node:timers/promises";
import test from "node:test";
import vm from "node:vm";

const sourcePath = process.env.OCR_CLIENT_SOURCE || new URL("../ocr_local/web/static/app.js", import.meta.url);
const source = await readFile(sourcePath, "utf8");
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

function element() {
  const listeners = new Map();
  return {
    value: "", textContent: "", hidden: true, disabled: false, checked: false, valid: true, files: [],
    addEventListener(name, handler) { listeners.set(name, handler); },
    emit(name, fields = {}) {
      const event = {defaultPrevented: false, preventDefault() {this.defaultPrevented = true;}, ...fields};
      listeners.get(name)?.(event);
      return event;
    },
    setAttribute(name, value) {this[name] = value;},
    replaceChildren(...children) {this.children = children;},
    reportValidity() {this.reports = (this.reports || 0) + 1; return this.valid;},
    checkValidity() {return this.valid;},
    click() {this.emit("click", {target: this});},
    classList: {add() {}, remove() {}},
  };
}

async function client({models = () => json(catalog), upload} = {}) {
  const elements = new Map();
  const document = element();
  document.getElementById = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  document.createElement = element;
  const el = id => document.getElementById(id);
  el("modelSettings").elements = ["modelSelect", "promptInput", "maxTokensInput", "maxPixelsInput",
    "quantizationInput", "forceInput", "resetSettingsBtn"].map(el);
  const calls = [];
  const streams = [];
  const context = vm.createContext({
    document, File, FormData,
    EventSource: class {
      constructor() {streams.push(this);}
      close() {this.closed = true;}
    },
    fetch: async (url, options = {}) => {
      calls.push({url, ...options});
      if (url === "/api/models") return models();
      if (url === "/api/upload") {
        return upload ? upload(options) : json({
          image_path: "/uploads/image.png", image_url: "/api/image/image.png",
          text: "", cached: false,
        });
      }
      if (url === "/api/recognize") return json({job_id: "test-job"});
      if (url === "/api/cached-result") return json({cached: false, text: ""});
      throw new Error("Неожиданный запрос: " + url);
    },
  });
  vm.runInContext(source, context, {filename: "app.js"});
  await settle();
  return {
    el, calls, streams,
    uploads: () => calls.filter(call => call.url === "/api/upload"),
    paste: items => document.emit("paste", {clipboardData: {items}}),
  };
}

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
  pending.resolve(json({image_path: "/first.png", image_url: "/first.png", text: ""}));
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
    return json({image_path: "/image.png", image_url: "/image.png", cached: true, text: settings.model_id});
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
    return json({image_path: "/retry.png", image_url: "/retry.png", text: ""});
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
