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
  return "Find useful, directly relevant sources for the user's topic. Match source types to the topic and the user's preferences: for practical software questions use maintained repositories, release notes, documentation, benchmarks and relevant community discussions; for academic questions seek research papers, DOI landing pages and accessible full-text PDF links. Prioritize recent, accessible original material, not generic official pages. Search broadly, but do not turn a practical question into a search for academic or official recognition. Prefer original source URLs over summaries, and when relevant distinguish full text from search snippets. Do not claim a source or study does not exist merely because this search did not find it. Report the useful findings and what each source contributes. Topic: " + topic;
}
