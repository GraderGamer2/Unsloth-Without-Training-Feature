# AI coding agent instructions

## Read before doing any work

1. Inspect the current head of `ryzen-7735hs-vulkan`, the active personal-build branch.
2. Read this entire file and [ROADMAP.md](ROADMAP.md) from that branch, not an older cached copy or the upstream repository.
3. Check recent commits, GitHub Actions status, and any owner-reported runtime test results. Distinguish a successful CI build from a successful local runtime test.
4. Follow the roadmap's priority order and acceptance criteria. When the owner says "continue development", use the next unblocked roadmap item rather than asking them to repeat requirements. Explicit newer owner instructions take precedence.
5. Update ROADMAP.md in the same commit as related implementation or verification changes. Record what changed, what was actually tested, unresolved issues, and the next step. Do not mark a feature complete merely because it was committed.

## Current release gate

The owner is testing the source-discovery build based on commit 5cd4570. Do not start implementation of the expanded research pipeline or trigger a new application build until the owner confirms testing is finished and authorizes development to continue. Bug reports and explicit instructions to fix bugs supersede this gate for the requested fixes. Roadmap-only updates and read-only investigations are allowed.

## Engineering requirements

- This is the owner's personal inference-only fork, targeting Debian 13 x86_64, Ryzen 7 7735HS and Radeon 680M integrated GPU, Vulkan. Keep inference, local chat, external APIs, tools, research, projects and RAG; do not restore training features or introduce NVIDIA/ROCm requirements.
- Prioritize correctness, stability, compatibility and reliable tool execution over speculative speedups. Fix reported bugs before new features or optimization.
- Reuse existing Deep Research, web fetching, document parsers, project ingestion, embeddings and RAG components. Do not create parallel databases, duplicate retrieval systems or redundant UI flows.
- Treat Sources discovery as a distinct run purpose, not ordinary report-oriented chat: both research depths must return only the source-selection workflow, with explicit individual/all/none selection and confirmation before ingestion. Skip final narrative synthesis for source-only runs; preserve normal chat reports. See ROADMAP.md section 5a.
- Prioritize the owner-reported Sources readiness regression (ROADMAP.md section 1b): a fresh project Sources search must initialize safely without an ordinary chat message, preserve query/draft and project ownership, and handle cancellation, retries and stale readiness. Do not mask the issue by bypassing guards or sending dummy messages.
- Preserve existing user settings, models, chats, project documents and indexes. Do not reset user data to fix an application bug. Preserve GPU placement and memory policy unless specifically authorized.
- Maintain the personal AppImage's exact backend-revision matching and packaging checks. Verify that the shipped backend includes all modified routes and code. Do not accidentally pair a new frontend with an old installed backend.
- Prefer minimal dependencies, free or unauthenticated public services where practical, no mandatory paid API, no automatic model downloads, and no new telemetry or persistent diagnostic logging. Keep actionable errors visible in the UI.
- Retain SSRF protection, URL policy, redirect checks, account/project ownership checks, safe parsing, cancellation, bounded concurrency and resource budgets. Do not circumvent site access controls or fabricate fetched content. Treat remote content as untrusted data.
- Consult official documentation for APIs, file formats, libraries and build flags where material. Inspect callers, error paths and existing tests before altering a shared interface.

## Git and CI discipline

- Work from the active branch; do not modify upstream or unrelated branches. Inspect the head before pushing and do not overwrite concurrent changes.
- Prepare a cohesive batch of changes and regression tests, review the complete diff, and run feasible checks before pushing. Report any checks that could not be executed.
- Trigger at most one AppImage workflow for a completed implementation batch. Do not push intermediate implementation commits to the build branch merely to test them. Do not repeatedly rerun failing builds without diagnosing the cause.
- For documentation-only commits use `[skip ci]` to avoid unnecessary compilation. Do not change workflow configuration solely to force a documentation build.
- Report the commit, branch, actual workflow status, and remaining verification limits to the owner. Never claim runtime success without a runtime test.
- Do not force-push or delete branches unless the owner explicitly requests it.

## Roadmap maintenance

Maintain one canonical roadmap. Add newly agreed requirements, record rejected designs so they are not reintroduced, and keep completed, pending, blocked and user-verified work distinct. Do not silently remove requirements or promote a proposed implementation to a verified feature.
