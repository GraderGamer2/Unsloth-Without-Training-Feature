// SPDX-License-Identifier: AGPL-3.0-only
import { Button } from "@/components/ui/button";
import { useChatProjects } from "@/features/chat/hooks/use-chat-projects";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import type { ResearchRun } from "@/features/chat/types/research";
import { toast } from "@/lib/toast";
import { useState } from "react";
import { discoverProjectSources, importProjectWebSource, type WebSourceCandidate } from "../api/rag-api";

function SourceImportList({ projectId, sources, researchRunId }: {
  projectId: string;
  sources: readonly WebSourceCandidate[];
  researchRunId?: string;
}) {
  const policy = useChatRuntimeStore((state) => state.researchWebsitePolicy);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [saved, setSaved] = useState<Set<string>>(() => new Set());
  const [failures, setFailures] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const unique = [...new Map(sources.map((source) => [source.url, source])).values()];
  const pending = unique.filter((source) => selected.has(source.url) && !saved.has(source.url));
  const save = async () => {
    if (busy || !pending.length) return;
    setBusy(true);
    try {
      // Sequential fetches limit peak RAM and avoid launching a burst of web
      // requests. Existing ingestion owns chunking and embedding on disk.
      for (const source of pending) {
        try {
          await importProjectWebSource(projectId, source, policy, researchRunId);
          setSaved((prev) => new Set(prev).add(source.url));
          setFailures((prev) => {
            const next = { ...prev };
            delete next[source.url];
            return next;
          });
        } catch (error) {
          setFailures((prev) => ({
            ...prev,
            [source.url]: error instanceof Error ? error.message : "Source could not be imported",
          }));
        }
      }
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="space-y-2">
      <div className="max-h-72 space-y-2 overflow-y-auto">
        {unique.map((source) => (
          <div key={source.url} className="rounded-md border border-border/60 p-2">
            <label className="flex items-start gap-2 text-xs">
              <input type="checkbox" className="mt-0.5 shrink-0"
                checked={selected.has(source.url) || saved.has(source.url)}
                disabled={busy || saved.has(source.url)}
                onChange={(event) => {
                  const checked = event.target.checked;
                  setSelected((prev) => {
                    const next = new Set(prev);
                    if (checked) next.add(source.url); else next.delete(source.url);
                    return next;
                  });
                }} />
              <span className="min-w-0 break-words font-medium">{source.title || source.url}</span>
            </label>
            <a className="block truncate text-xs text-muted-foreground underline"
              href={source.url} target="_blank" rel="noopener noreferrer" title={source.url}>{source.url}</a>
            {source.snippet ? <p className="mt-1 line-clamp-3 text-xs text-muted-foreground">{source.snippet}</p> : null}
            {saved.has(source.url) ? <p className="text-xs text-muted-foreground">Added · indexing status in Sources</p> : null}
            {failures[source.url] ? <p role="alert" className="break-words text-xs text-destructive">{failures[source.url]}</p> : null}
          </div>
        ))}
      </div>
      <Button type="button" size="sm" variant="outline" disabled={busy || pending.length === 0}
        onClick={() => void save()}>{busy ? "Importing sources…" : `Add selected (${pending.length})`}</Button>
      <p className="text-xs text-muted-foreground">Original accessible text is fetched and indexed, not the search snippets or AI report. Website extracts may be incomplete.</p>
    </div>
  );
}

export function WebSourceCollector({ projectId }: { projectId: string }) {
  const policy = useChatRuntimeStore((state) => state.researchWebsitePolicy);
  const [query, setQuery] = useState("");
  const [sources, setSources] = useState<WebSourceCandidate[] | null>(null);
  const [searchId, setSearchId] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async () => {
    const value = query.trim();
    if (!value || busy) return;
    setBusy(true);
    setError(null);
    setSources(null);
    try {
      if (/^https?:\/\//i.test(value)) {
        await importProjectWebSource(projectId, { url: value, title: value.slice(0, 200) }, policy);
        toast.success("Website added to project sources. Indexing will run automatically.");
        setQuery("");
      } else {
        const result = await discoverProjectSources(projectId, value, policy);
        setSources(result.sources);
        setSearchId((id) => id + 1);
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "Source discovery failed");
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="mb-3 space-y-2 rounded-xl border border-border/60 p-3">
      <h3 className="text-sm font-semibold">Find sources</h3>
      <p className="text-xs text-muted-foreground">Ask for relevant websites, or paste a URL to add directly.</p>
      <form className="flex flex-col gap-2" onSubmit={(event) => { event.preventDefault(); void submit(); }}>
        <input type="text" value={query} onChange={(event) => setQuery(event.target.value)}
          aria-label="Source discovery query or website URL" maxLength={4096} disabled={busy}
          placeholder="e.g. research papers about Vulkan inference"
          className="min-w-0 rounded-md border bg-background px-2 py-2 text-sm" />
        <Button type="submit" size="sm" variant="outline" disabled={!query.trim() || busy}>
          {busy ? "Working…" : /^https?:\/\//i.test(query.trim()) ? "Add website" : "Discover sources"}
        </Button>
      </form>
      {error ? <p role="alert" className="break-words text-xs text-destructive">{error}</p> : null}
      {sources?.length ? <SourceImportList key={searchId} projectId={projectId} sources={sources} /> :
        sources ? <p className="text-xs text-muted-foreground">No accessible sources found. Try another query.</p> : null}
    </section>
  );
}

export function ResearchSourceCollector({ run, defaultProjectId }: {
  run: ResearchRun;
  defaultProjectId: string | null;
}) {
  const { projects } = useChatProjects();
  const [projectId, setProjectId] = useState(defaultProjectId ?? "");
  const [expanded, setExpanded] = useState(false);
  const target = projects.some((project) => project.id === projectId) || projectId === defaultProjectId ? projectId : "";
  return (
    <section className="my-3 space-y-2 rounded-xl border border-border/60 p-3">
      <Button type="button" variant="outline" size="sm" aria-expanded={expanded}
        onClick={() => setExpanded((prev) => !prev)}>
        {expanded ? "Hide source collection" : "Add research sources to project"}
      </Button>
      {expanded ? (
        <>
          <label className="block text-xs text-muted-foreground">Destination project
            <select value={target} onChange={(event) => setProjectId(event.target.value)}
              className="mt-1 block w-full rounded-md border bg-background p-2 text-sm">
              <option value="">Choose an existing project</option>
              {defaultProjectId && !projects.some((project) => project.id === defaultProjectId) ?
                <option value={defaultProjectId}>Current project</option> : null}
              {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
            </select>
          </label>
          {target ? <SourceImportList key={target} projectId={target} sources={run.sources} researchRunId={run.id} /> :
            <p className="text-xs text-muted-foreground">Create or select a project to retain these sources independently of this chat.</p>}
        </>
      ) : null}
    </section>
  );
}
