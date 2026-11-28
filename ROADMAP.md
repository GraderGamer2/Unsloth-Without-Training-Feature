# Personal Unsloth development roadmap

Active branch: `ryzen-7735hs-vulkan`  
Owner-approved requirements recorded: 2026-11-28  
Read [AGENTS.md](AGENTS.md) before implementation.

## 1. Current status and release gate

- **Owner testing pending:** source-discovery build based on `5cd4570e2dc9bca5c8e33296e9ac4aa52ea39fe3`. Do not implement the next batch until the owner finishes testing and explicitly continues development.
- **Owner runtime-confirmed on the preceding build:** research now produced a readable report rather than raw function invocations; source import reached the backend. The Minecraft 1.12.2 test discovered 16 sources, submitted six, and five contained actual source text. The sixth was a Reddit bot challenge page. This is not proof that all research and import paths work.
- **Implemented, awaiting local verification in the current build:** reject common verification pages; preserve original direct PDF and supported arXiv downloads; improve topic-appropriate search prompts; show chat readiness; distinguish indexing completion from submission and permit failed-index retries. Current original PDF fetch safety cap: 32 MiB, separate from the normal local upload cap. No arbitrary total project-source limit added.
- **Known outstanding issues:** project Sources search may wait 1–2 minutes for chat runtime initialization; showing the readiness state does not fix that latency. Web extraction is incomplete on some sites and can fail due to bot protection, access restrictions, extraction errors or network failures. General open-access PDF resolution, non-PDF original remote document imports, and automatic linked-source discovery are pending.
- **Prior bug fixes to preserve:** reject raw synthesis tool calls and structured actions; retry synthesis using existing evidence; retain collected sources after failed/stopped research; avoid replacing a valid partial report with malformed output; ensure AppImage/frontend/backend revision compatibility; avoid HTTP 405 caused by stale backend. Keep earlier tested performance work; do not assume all optimizations are independently benchmarked.

## 1a. High-priority update/install regression (owner report, 2026-11-28)

- **Pending fix; owner reports:** opening each replacement AppImage now triggers downloads/setup, with over 1 GB of uv-related storage and Node and uncompiled-module entries. Exact disk paths and downloaded bytes are not yet measured.
- **Confirmed code path:** unique backend revision stamping (introduced in 2255f7f) causes managed preflight to classify a different installed revision as stale; the normal repair path runs install.sh, which installs packages and invokes studio/setup.sh, including Node/dependency setup checks. The installed revision must still match the shipped backend; do not revert that protection or reintroduce stale API behavior.
- **Required fix:** distinguish a healthy environment with only a bundled-backend revision mismatch from an actual missing/broken environment. For the former, update/downgrade only the bundled local wheel in the existing venv, offline and without dependency resolution or unnecessary uv cache copies, after checking ownership, process shutdown, Python compatibility and dependency manifest compatibility. Verify the installed revision and required capabilities before startup. Full installation/repair is reserved for initial setup, missing/corrupt dependencies, or explicit repair; clearly explain when network access is needed. Use existing locking and safe failure/rollback mechanisms; never delete settings, models, documents or indexes.
- Audit whether Node/npm is required for specific retained tools (not for the precompiled desktop UI); do not repeatedly provision it on backend-only updates, and do not remove required runtimes blindly. Audit uv/npm cache growth and generated-module paths; prefer safe, scoped cleanup of obsolete application-owned cache entries, never shared caches or user assets without authorization.
- Acceptance: upgrade and downgrade between two different bundled backend versions with a healthy installed environment and networking disabled; no dependency downloads, no Node setup, no full setup.sh pass, no unnecessary cache growth; new backend routes present and healthy; all user data preserved. Test initial install and genuine repair separately. No application code change or build has been initiated by recording this issue.

## 1b. High-priority Sources readiness regression (owner runtime report, 2026-11-28)

- **Pending fix; reproduced by owner:** on the latest build, opening a project's Sources search without first using ordinary chat leaves the control at "Preparing chat…" or produces a readiness toast. Sending a message in an ordinary chat (including the project's new-chat composer) and returning to Sources makes search work. The user must never have to send a dummy message to unlock Sources.
- **Code inspection:** `studio/frontend/src/features/chat/chat-page.tsx` gates Sources on `active && runtimeReady && !pendingNewThreadId`. `projectRuntimeReady` is keyed by project ID and set via `onInitialHistoryReady`; `runtime-provider.tsx` signals this from the history adapter's load completion. `startSourceResearch` also requires a mounted composer form and submits via a double animation-frame callback with project ownership checks. These are confirmed dependencies, not proof of the exact event sequence on the owner's machine. The latest visible readiness label did not resolve initialization.
- **Required fix:** explicitly initialize or await the actual project-scoped research/composer readiness on the Sources path, including a fresh empty project, independently of whether ordinary chat history has loaded or a message has been sent. Prefer reusing the same runtime/run orchestration with a purpose-specific entry point. Preserve queries while preparing; bind readiness and queued work to the current project and session/nonce, invalidate stale events, handle cancellation and switching, reject duplicate runs, and expose genuine initialization errors with retry. Do not merely remove safety checks or generate dummy messages, extra chats, model downloads, or unnecessary model reloads.
- Acceptance: on a cold AppImage start, with a configured/loaded model, open an existing or empty project and immediately search Sources without using any ordinary chat; the request starts once when ready, remains project-scoped, and never hangs indefinitely. Test warm startup, failed initialization/retry, rapid repeated submission, switching projects during preparation, cancellation and a preexisting ordinary chat draft. Preserve drafts, existing sources and user data. No code fix or application build has yet been performed for this report.

## 2. Priority and workflow

1. Resolve owner-reported runtime bugs and regressions first, including the repeated AppImage setup/download regression in section 1a and the Sources readiness regression in section 1b. Diagnose rather than merely hide symptoms.
2. Verify the current build with the owner's feedback, including readiness, report synthesis, source import and PDF behavior.
3. Implement the research expansion below as a coherent, tested batch, reusing existing architecture.
4. Resume broader performance work only after research and source ingestion are reliable.

Maintain this roadmap with actual progress and verification evidence after each batch. A successful build is not equivalent to a successful test on the owner's machine.

## 3. Research modes — exactly two

Both the main chat and the project Sources search bar expose:
- **Deep Research (default):** thorough, targeted discovery and verification with reasonable resource usage; follow relevant links and acquire useful documents.
- **Exhaustive Research:** substantially broader discovery, recursive exploration, more follow-up queries and source verification, until evidence coverage and diminishing returns justify stopping, the configured resource budget is reached, or the user stops it.

No Fast Research mode. Do not silently reduce Deep Research to a shallow search.

Both modes use automatic query expansion, complementary search providers, academic discovery where relevant, relevant linked-source discovery, source deduplication, extraction, semantic relevance filtering and evidence verification. Exhaustive mode must not stop merely because a small model emits a finish action; independently evaluate progress, missing dimensions and new relevant evidence. Keep time/network/memory limits configurable and transparent; distinguish per-run safety budgets from persistent source capacity. No arbitrary new cap on total project sources.

## 4. Research orchestration and quality

- Let the selected LLM identify the question, research dimensions and search intent, but use deterministic backend logic for discovery, fetching, validation, deduplication, and progress/coverage checks where possible. Make the pipeline useful with small local models (e.g. MiniCPM or Spark); no mandatory larger model.
- Use topic-appropriate sources. For practical software questions prioritize maintained projects, release notes, source documentation, benchmarks and relevant community posts; do not force academic publications or official endorsement. For academic questions discover papers, DOI metadata, publisher/repository pages and legitimate open full text.
- Explore multiple complementary public search providers. Evaluate OpenAlex, Crossref, Unpaywall and other appropriate open academic services, respecting their current authentication, attribution, usage and rate-limit requirements. No mandatory paid API or user-secret exposure.
- Follow relevant links recursively, discover downloadable documents and useful media, and pursue missing dimensions. Rank by relevance, recency where requested, source quality, accessibility and evidence coverage, not merely authority or search-result order.
- Use the existing configured embedding model (the owner uses Granite; confirm exact variant and capabilities) for semantic relevance and retrieval where feasible. Keep lexical matching and provenance as complementary signals. Avoid embedding irrelevant candidates or re-embedding unchanged data.
- Deduplicate normalized URLs and actual content while preserving useful provenance; avoid repeated visits and loops. Respect query privacy and website access policy throughout, including redirects and discovered links.
- On access denial, bot protection or failed extraction, seek legitimate accessible alternatives, such as public APIs, project repositories, open repositories or author copies. Do not bypass paywalls, authentication or bot protection.
- Distinguish discovered links, snippets, actual fetched content, extraction failures and indexed documents. Never present a snippet, challenge page, AI summary or fabricated text as the original source. Report coverage gaps honestly, not unsupported claims that sources do not exist.
- Preserve cancellation, partial findings, source import after failed/stopped runs, and bounded synthesis recovery. Invalid tool markup must never be marked as a completed research report.

## 5. Source discovery and original file ingestion

Extend the existing Unsloth project Sources panel and Deep Research engine, not a second NotebookLM-style architecture. A user enters a topic, researches, reviews discovered sources and chooses which become permanent project sources. Allow additional imports across runs, manual additions/removals using existing controls, retry of failed imports, and no new project-wide source-count quota.

### 5a. Dedicated source-discovery experience (owner requirement, 2026-11-28)

- **Sources search is not ordinary Deep Research chat.** Both Deep Research and Exhaustive Research remain available as discovery-depth choices, sharing the same safe discovery, fetch, verification and ranking engine. Distinguish run purpose (source discovery versus report-oriented chat) from research depth.
- A query submitted from a project's Sources search bar stays in the Sources workflow: show concise progress/cancel controls while searching and, on completion, a source-selection prompt/list only. Do not output a normal research report, long narrative synthesis, or ordinary chat transcript; do not require a separate chat or plan-approval conversation merely to discover sources. Internal planning and coverage checks may still run. Skip final report synthesis entirely for source-discovery runs rather than generate it and hide it.
- The result lists deduplicated relevant sources with title, URL, type and availability/extraction status; concise relevance metadata is acceptable, not an essay. Provide individual checkboxes, Select all, Deselect all, Keep selected, and Discard all/close without saving. Only explicitly confirmed selections are imported via existing native project ingestion/RAG; unselected candidates are not persisted as project sources. Report actual indexing success/failure and allow retries. Do not silently add sources or remove existing ones when discarding candidates.
- Keep the results and selection scoped to the originating project/run, including partial results on cancellation/failure where available; preserve drafts and other chats. No unwanted ordinary chat threads or messages as a side effect. Reuse existing research-run state, permission checks, ingestion and storage; no parallel database or RAG pipeline.
- Main-chat Deep Research and Exhaustive Research remain report-oriented and retain their existing ability to optionally import discovered sources. Do not remove or degrade normal chat research to implement Sources discovery.
- Acceptance: from a cold project landing, submit a query in each mode; show only progress followed by the source-selection UI, without ordinary report generation or dummy chat. Verify keep-all, arbitrary subset, deselect-all/discard, failed/partial search, indexing retry and project-switch isolation; normal chat still produces a report. Verify no report-synthesis model call occurs for source-only runs and no automatic permanent imports occur. Pending implementation and local runtime verification.

Support relevant accessible formats through existing native parsers and ingestion where possible:
- HTML/webpages: extract useful main content, tables and relevant links; remove navigation/ads/challenge pages.
- PDF and academic papers: discover legitimate full-text links (including DOI and publisher landing pages), download original documents and preserve page-aware parsing/OCR.
- Word DOC/DOCX; Excel XLS/XLSX; PowerPoint PPT/PPTX; CSV/TSV; OpenDocument ODT/ODS/ODP; EPUB; plain text/Markdown; JSON/XML and relevant code/config files. Preserve native document structure, sheets/tables/slides/headings and notes where supported by the parser.
- GitHub: relevant README, release notes, documentation, issues and source files; community discussions: actual posts and relevant replies via permitted access.
- Relevant images, charts, diagrams and screenshots: preserve originals and process according to available capabilities (section 6).

Detect actual format by safe content signatures and MIME type, not URL extension alone. Reuse safe download/redirect and native upload/parse limits; support configurable bounded downloads and avoid unnecessary memory copies. Preserve source URL, title, retrieval time, content type and original file association where feasible. Original downloads should go through the same ingestion and embedding path as local uploads. Clearly disclose unsupported formats and partial extraction.

Keep source collection separate from permission to persist into a project: user-selected sources are imported; discovery alone must not silently populate the permanent knowledge base. Temporary research evidence must remain usable without creating unwanted permanent documents.

## 6. Images, OCR, vision and embedding capabilities

Detect capabilities of the actual loaded/configured models and available local OCR tools, rather than assuming a model can process images.
- Vision-capable model: extract useful visual descriptions, charts, diagram relationships and visible text where supported.
- Local OCR available: extract visible text even with a text-only chat model; make clear when only OCR, not visual interpretation, was performed.
- Compatible multimodal embedding model AND retrieval support: use image embeddings when the existing architecture genuinely supports them. A standard Granite text embedding model can index extracted text/descriptions, not raw pixels.
- No usable vision or OCR: warn explicitly; preserve/download the original image if accessible and safe, label it as downloaded but not content-indexed. Never invent an image description or imply successful OCR.
- Apply equivalent checks to scanned PDFs, embedded images and document charts. Preserve readable text even if visual components cannot be interpreted.
- Permit reprocessing later with a compatible model. Never download additional AI models without the owner's permission. An unsupported image must not fail the entire run.

Show meaningful per-source states: discovered, downloading, extracted, submitted/indexing, indexed, OCR-only, downloaded/not indexed, inaccessible, unsupported, failed/retryable. Preserve originals and source provenance; distinguish a stored image from retrievable evidence.

## 7. Stability, efficiency and compatibility

- Fix the project-home readiness bottleneck by tracing runtime initialization and safe composer submission; do not simply bypass ownership/readiness checks. Preserve the entered query and show the actual reason a search cannot start.
- Bound concurrent network requests, parsing, OCR and indexing based on available resources. Respect cancellation and provider throttling, and avoid runaway recursive crawling.
- Reuse existing project database, retrieval, embeddings, native parsers, web safety and document lifecycle. No duplicate RAG or separate research database.
- Preserve current user settings, model cache, projects, documents and indexes across updates. No GPU placement/memory-policy changes, mandatory paid services, extra model downloads, telemetry or needless dependency growth.
- Keep the personal AppImage compatible with the owner's Ryzen 7 7735HS / Radeon 680M Vulkan target. Keep build-time backend revision stamping and shipped-file checks.
- Do not restore training or reintroduce the earlier reverted standalone NotebookLM-like retrieval/evidence-packing implementation. The owner wants integration with the existing project architecture.

## 8. Acceptance tests for the next implementation batch

Add automated regression coverage and report feasible test results before one consolidated build. Include:
- Source search initialization from a cold launch without ordinary chat; safe project-scoped submission, failure/retry, cancellation, query preservation and project switching.
- Dedicated Sources-only output and explicit keep/discard workflow in both modes (section 5a); skip report synthesis, no unwanted chat creation, select all/subset/none, no automatic import, normal chat report behavior unchanged.
- Research synthesis with a small model, malformed tool output, successful recovery and failed/stopped source retention.
- Practical community/software research (e.g. Minecraft 1.12.2 optimization) without irrelevant academic/official bias.
- Academic research discovering accessible full-text PDFs; verified native indexing of PDF, DOCX, XLSX and other supported formats including misleading URL extensions/MIME types.
- Image handling with vision, OCR-only, and neither capability; no fabricated indexing success.
- Actual Reddit-style challenge-page fixture, blocked or rate-limited fetches, permitted alternatives and explicit failures.
- Duplicate sources, recursive cycles, bounded resource use, cancellation, project isolation, SSRF and redirect-policy checks.
- Original files and provenance preserved; indexing status accurately reflects completion or failure; previously stored sources remain intact.
- Both modes available and distinct; Exhaustive mode does not terminate solely on a premature model finish action.
- Package and frontend/backend integration checks, then owner runtime testing with actual local models and embeddings.

## 9. Next action

Await owner confirmation of current build testing. Fix any newly reported regressions first. Then implement the expanded two-mode research pipeline in dependency order (discovery and safe acquisition, native ingestion/capability checks, coverage orchestration and UI), with a single consolidated workflow when the batch is ready. Update this file with actual results instead of relying on previous chat context.
