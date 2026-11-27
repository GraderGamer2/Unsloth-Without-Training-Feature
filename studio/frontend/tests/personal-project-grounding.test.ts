import { strict as assert } from "node:assert";
import { test } from "node:test";
import { packGroundedEvidence, finishGroundedAnswer } from "../src/features/rag/api/project-grounding.ts";

test("evidence is bounded, unique and references only actual passages", () => {
  const row = { chunkId: "one", documentId: "d", filename: "Original", page: 2,
    text: "Bukti asli. ".repeat(1000), sourceUrl: "https://example.org/p", sourcePartial: true };
  const packet = packGroundedEvidence([row, row], 2048, "Apa buktinya?");
  assert.equal(packet.passages.length, 1);
  assert.equal(JSON.parse(packet.evidence).id, "S1");
  assert.ok(new TextEncoder().encode(packet.evidence).length < 3072);
  const answer = finishGroundedAnswer("Bukti [S1], unknown [S9].", packet.passages);
  assert.match(answer, /unverified source reference/);
  assert.match(answer, /Original, page 2/);
  assert.match(answer, /https:\/\/example.org\/p/);
  assert.doesNotMatch(answer, /\[S9\]/);
  assert.match(finishGroundedAnswer("Uncited", packet.passages), /No valid source citations/);
});
