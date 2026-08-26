const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const expectedCitation = `@misc{denglab2026streamwam,
  title        = {Stream-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation},
  author       = {{DENG Lab}},
  year         = {2026},
  howpublished = {Project page},
  organization = {Shanghai Jiao Tong University},
  url          = {https://sjtu-deng-lab.github.io/StreamWAM/}
}`;
const script = fs.readFileSync(path.join(__dirname, "..", "docs", "script.js"), "utf8");

async function runScenario({ clipboardAvailable }) {
  let clickHandler = null;
  let clipboardText = null;
  let fallbackText = null;
  let fallbackCommand = null;
  let appendedTextarea = null;
  const timers = [];
  const copyButton = {
    textContent: "Copy",
    addEventListener(type, handler) {
      if (type === "click") clickHandler = handler;
    },
  };
  const citation = { textContent: `\n${expectedCitation}\n` };
  const document = {
    documentElement: { classList: { add() {} } },
    body: {
      appendChild(element) {
        appendedTextarea = element;
      },
      removeChild(element) {
        assert.equal(element, appendedTextarea);
      },
    },
    querySelector(selector) {
      if (selector === ".citation-copy") return copyButton;
      if (selector === "#citation-bibtex") return citation;
      return null;
    },
    createElement(tag) {
      assert.equal(tag, "textarea");
      return {
        value: "",
        style: {},
        setAttribute() {},
        select() {
          fallbackText = this.value;
        },
      };
    },
    execCommand(command) {
      fallbackCommand = command;
      return true;
    },
  };
  const navigator = clipboardAvailable
    ? { clipboard: { async writeText(text) { clipboardText = text; } } }
    : {};

  vm.runInNewContext(script, {
    document,
    navigator,
    setTimeout(callback, delay) {
      timers.push({ callback, delay });
    },
  });

  assert.equal(typeof clickHandler, "function", "script.js must register the citation-copy click handler");
  await clickHandler();
  assert.equal(copyButton.textContent, "Copied");
  assert.equal(timers.length, 1);
  assert.equal(timers[0].delay, 1800);
  timers[0].callback();
  assert.equal(copyButton.textContent, "Copy");

  if (clipboardAvailable) {
    assert.equal(clipboardText, expectedCitation);
    assert.equal(fallbackCommand, null);
  } else {
    assert.equal(fallbackText, expectedCitation);
    assert.equal(fallbackCommand, "copy");
  }
}

(async () => {
  await runScenario({ clipboardAvailable: true });
  await runScenario({ clipboardAvailable: false });
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
