// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";
import { projectSourceResearchQuestion } from "../src/features/chat/utils/project-source-research.ts";

test("source search is explicit, project scoped, and only the latest user message", () => {
  const user = (text: string) => ({ role: "user", content: [{ type: "text", text }] });
  const prompt = user("/find-sources research about Vulkan");
  assert.match(projectSourceResearchQuestion([prompt], "p", false) ?? "", /research about Vulkan/);
  assert.equal(projectSourceResearchQuestion([prompt], null, false), null);
  assert.equal(projectSourceResearchQuestion([prompt], "p", true), null);
  assert.equal(projectSourceResearchQuestion([prompt, user("normal chat")], "p", false), null);
  assert.equal(projectSourceResearchQuestion([{ ...prompt, role: "assistant" }], "p", false), null);
  assert.equal(projectSourceResearchQuestion([{ ...prompt, attachments: [{}] }], "p", false), null);
  assert.equal(projectSourceResearchQuestion([user("/find-sources   ")], "p", false), null);
});
import { fileImportSource, readAllText } from "../src/features/chat/utils/json-record-stream.ts";
import { startVisibilityPoll } from "../src/features/hub/download-manager/visibility-poll.ts";

test("early file import termination cancels the stream and releases its reader", async () => {
  let cancelled = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      controller.enqueue(new Uint8Array([97, 98, 99]));
    },
    cancel() {
      cancelled++;
    },
  });
  const file = { name: "test.json", size: 10, stream: () => stream } as unknown as File;
  await assert.rejects(readAllText(fileImportSource(file), 2, "JSON"), /too large/);
  assert.equal(cancelled, 1);
  assert.equal(stream.locked, false);
});

test("failed download polling re-arms once and stop removes the timer", async () => {
  const document = Object.assign(new EventTarget(), { hidden: false });
  const timers = new Map<number, () => void>();
  let id = 0;
  let calls = 0;
  const poll = startVisibilityPoll(async () => {
    if (++calls === 1) throw new Error("temporary network failure");
    return false;
  }, () => 100, {
    targets: {
      document,
      setTimeout(callback) {
        timers.set(++id, callback);
        return id;
      },
      clearTimeout(handle) {
        timers.delete(handle as number);
      },
    },
  });
  poll.refresh();
  await Promise.resolve();
  assert.equal(timers.size, 1);
  assert.equal(calls, 1);
  poll.refresh();
  await Promise.resolve();
  assert.equal(calls, 2);
  assert.equal(timers.size, 0);
  poll.stop();
  document.dispatchEvent(new Event("visibilitychange"));
  assert.equal(calls, 2);
});
