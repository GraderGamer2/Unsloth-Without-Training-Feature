// SPDX-License-Identifier: AGPL-3.0-only
import { Button } from "@/components/ui/button";
import { cancelResearchRun, retryResearchRun, createProjectResearchRun, discardProjectResearchRun, followResearchRun, latestProjectResearchRun } from "@/features/chat/api/research-api";
import { loadConnectionsEnabled, loadExternalProviders, parseExternalModelId, providerModelSupportsStudioTools } from "@/features/chat/external-providers";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import type { ResearchDepth, ResearchRun } from "@/features/chat/types/research";
import { useCallback, useEffect, useRef, useState } from "react";
import { pinnedCheckpoint, servedCheckpoint } from "@/features/hub/lib/model-identity";
import { ResearchSourceImport } from "./research-source-import";

const settled = (run: ResearchRun) => ["completed", "failed", "cancelled"].includes(run.status);
function message(error: unknown) { return error instanceof Error ? error.message : String(error); }

export function ProjectSourceResearch({projectId}: {projectId: string}) {
  const [query, setQuery] = useState("");
  const [depth, setDepth] = useState<ResearchDepth>("deep");
  const [run, setRun] = useState<ResearchRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const live = useRef(false);
  const requestBusy = useRef(false);
  const follower = useRef<AbortController | null>(null);
  const runRef = useRef<ResearchRun | null>(null);
  const attach = useCallback((initial: ResearchRun | null) => {
    follower.current?.abort();
    runRef.current = initial;
    setRun(initial);
    if (!initial || settled(initial)) return;
    const controller = new AbortController();
    follower.current = controller;
    void (async () => {
      try {
        for await (const update of followResearchRun(initial.id, {initialRun: initial, signal: controller.signal})) {
          if (!controller.signal.aborted && live.current && update.run.projectId === projectId) {
            runRef.current = update.run;
            setRun(update.run);
          }
        }
      } catch (err) {
        if (!controller.signal.aborted && live.current) setError(message(err));
      }
    })();
  }, [projectId]);
  useEffect(() => {
    live.current = true;
    const controller = new AbortController();
    void latestProjectResearchRun(projectId, controller.signal).then(current => {
      if (!controller.signal.aborted) attach(current);
    }).catch(err => {
      if (!controller.signal.aborted) setError(message(err));
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false);
    });
    return () => { live.current = false; controller.abort(); follower.current?.abort(); };
  }, [projectId, attach]);
  async function start() {
    if (!query.trim() || requestBusy.current || (runRef.current && !settled(runRef.current))) return;
    requestBusy.current = true;
    setBusy(true);
    setError("");
    try {
      const state = useChatRuntimeStore.getState();
      if (state.incognito) throw new Error("Project source research is persistent. Exit incognito first.");
      const checkpoint = state.params.checkpoint;
      if (!checkpoint) throw new Error("Select and load a model before starting source research.");
      const external = parseExternalModelId(checkpoint);
      const provider = external && loadConnectionsEnabled() ? loadExternalProviders().find(p => p.id === external.providerId) : undefined;
      if (external && (!provider || providerModelSupportsStudioTools(provider.providerType, external.modelId) !== true)) {
        throw new Error("Select an enabled connection that supports Studio research tools.");
      }
      if (runRef.current && settled(runRef.current)) {
        await discardProjectResearchRun(projectId, runRef.current.id);
        if (live.current) attach(null);
      }
      const created = await createProjectResearchRun(projectId, {
        question: query.trim(), depth,
        inferenceRequest: {model: external?.modelId ?? pinnedCheckpoint(servedCheckpoint(state.loadedModels, checkpoint, state.activeGgufVariant), state.activeGgufVariant),
          temperature: state.params.temperature, topP: state.params.topP, maxTokens: Math.max(1, Math.min(8192, state.params.maxTokens || 2048)),
          ...(external && provider ? {providerId: provider.id, providerType: provider.providerType, externalModel: external.modelId} : {})},
        budgets: {modelTimeoutSeconds: state.researchModelTimeoutSeconds},
        websitePolicy: state.researchWebsitePolicy,
      });
      if (live.current) { attach(created); setQuery(""); }
    } catch (err) {
      if (live.current) {
        setError(message(err));
        // A request accepted before disconnect may already have created a durable run.
        try { const latest = await latestProjectResearchRun(projectId); if (live.current && latest) attach(latest); } catch { /* show original failure */ }
      }
    } finally { requestBusy.current = false; if (live.current) setBusy(false); }
  }
  async function stop() {
    if (!runRef.current || settled(runRef.current)) return;
    try { const stopped = await cancelResearchRun(runRef.current.id); if (live.current) attach(stopped); }
    catch (err) { if (live.current) setError(message(err)); }
  }
  async function retry() {
    const current = runRef.current;
    if (!current || current.status !== "failed" || requestBusy.current) return;
    requestBusy.current = true;
    setBusy(true);
    setError("");
    try {
      const state = useChatRuntimeStore.getState();
      const retried = await retryResearchRun(current.id, {modelTimeoutSeconds: state.researchModelTimeoutSeconds, websitePolicy: state.researchWebsitePolicy});
      if (live.current) attach(retried);
    } catch (err) { if (live.current) setError(message(err)); }
    finally { requestBusy.current = false; if (live.current) setBusy(false); }
  }
  async function discard() {
    const current = runRef.current;
    if (!current || !settled(current) || requestBusy.current) return;
    requestBusy.current = true;
    setBusy(true);
    try { await discardProjectResearchRun(projectId, current.id); if (live.current) attach(null); }
    catch (err) { if (live.current) setError(message(err)); }
    finally { requestBusy.current = false; if (live.current) setBusy(false); }
  }
  const running = run && !settled(run);
  const coverage = run?.steps.slice().reverse().find(step => step.result?.coverage)?.result?.coverage;
  return <div className="mb-4 rounded-2xl border border-border bg-muted/30 p-4">
    <form onSubmit={event => {event.preventDefault(); void start();}}>
      <label htmlFor="project-source-research" className="text-sm font-medium">Find sources</label>
      <textarea id="project-source-research" rows={2} maxLength={1600} value={query} onChange={event => setQuery(event.target.value)}
        placeholder="What would you like to research?" className="mt-2 w-full resize-y rounded-xl bg-background p-3 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"/>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <select aria-label="Research depth" className="rounded-md border bg-background p-2 text-sm" value={depth} disabled={Boolean(running) || busy}
          onChange={event => setDepth(event.target.value === "exhaustive" ? "exhaustive" : "deep")}>
          <option value="deep">Deep Research</option><option value="exhaustive">Exhaustive Research</option>
        </select>
        <Button type="submit" size="sm" disabled={loading || busy || Boolean(run) || !query.trim()}>{loading ? "Loading…" : busy ? "Starting…" : "Search sources"}</Button>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">Searches without creating a chat or report. Deep: up to 12 steps / 40 candidates; Exhaustive: up to 30 steps / 100 candidates. Nothing is permanently added until you select it.</p>
    </form>
    {error && <p role="alert" className="mt-2 text-sm text-destructive">{error}</p>}
    {run && <div className="mt-3 space-y-2">
      <p role="status" className="text-sm">{running ? `${run.status === "planning" ? "Planning search" : run.status === "cancelling" ? "Stopping" : "Discovering sources"} · ${run.sources.length} candidates` : `${run.status === "completed" ? "Search completed" : run.status === "cancelled" ? "Search stopped" : "Search failed"} · ${run.sources.length} candidates`}</p>
      {coverage && <p className="text-xs text-muted-foreground">{coverage.downloaded} acquired · {coverage.skippedExisting} existing or duplicate skipped · {coverage.distinctDomains} domains{coverage.uncoveredTerms.length ? ` · Unmatched topic terms: ${coverage.uncoveredTerms.join(", ")}` : ""}</p>}
      {running && <Button type="button" size="sm" variant="outline" disabled={run.status === "cancelling"} onClick={() => void stop()}>Stop search</Button>}
      {run.error && <p className="text-xs text-destructive">{run.error}</p>}
      {run.status === "failed" && <Button type="button" size="sm" variant="outline" disabled={busy} onClick={() => void retry()}>Retry search</Button>}
      {!running && (run.sources.length ? <ResearchSourceImport key={run.id} run={run} onDiscard={() => void discard()}/> :
        <><p className="text-xs text-muted-foreground">No new candidates were found. Existing project sources are excluded.</p><Button type="button" size="sm" variant="outline" disabled={busy} onClick={() => void discard()}>New search</Button></>)}
    </div>}
  </div>;
}
