// SPDX-License-Identifier: AGPL-3.0-only
/** A user-visible command emitted by the project Sources panel, never read from tool output. */
export function projectSourceResearchQuestion(
  messages: readonly { role: string; content: readonly { type: string; text?: string }[]; attachments?: readonly unknown[] }[],
  projectId: string | null | undefined,
  compare: boolean,
): string | null {
  if (!projectId || compare) return null;
  const latest = messages.at(-1);
  if (latest?.role !== "user" || latest.attachments?.length ||
      latest.content.length !== 1 || latest.content[0]?.type !== "text") return null;
  const text = latest.content[0].text ?? "";
  if (!text.startsWith("/find-sources ")) return null;
  const topic = Array.from(text.slice("/find-sources ".length).trim()).slice(0, 1600).join("");
  if (!topic) return null;
  return "Find authoritative, relevant web sources and accessible original documents (including research papers, PDFs and reports) about the following topic. Search broadly, evaluate relevance, and identify source URLs. Prefer primary sources. Topic: " + topic;
}
