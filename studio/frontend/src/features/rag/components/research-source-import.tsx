// SPDX-License-Identifier: AGPL-3.0-only
import { Button } from "@/components/ui/button";
import { getStoredChatThread } from "@/features/chat/utils/chat-history-storage";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import type { ResearchRun } from "@/features/chat/types/research";
import { useEffect, useMemo, useRef, useState } from "react";
import { announceProjectSourcesUpdated, importResearchSource, invalidateProjectSources } from "../api/rag-api";
import { watchIngestion } from "../api/save-markdown-source";

export function ResearchSourceImport({ run }: { run: ResearchRun }) {
  const [projectId, setProjectId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const sources = useMemo(() => Array.from(new Map(run.sources.map(s => [s.url, s])).values()), [run.sources]);
  const [selected, setSelected] = useState<Set<string>>(() => new Set(sources.map(s => s.url)));
  const [queued, setQueued] = useState<Set<string>>(new Set());
  const [indexed, setIndexed] = useState<Set<string>>(new Set());
  const [existing, setExisting] = useState<Set<string>>(new Set());
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const active = useRef(false);
  const running = useRef(false);

  useEffect(() => {
    active.current = true;
    let cancelled = false;
    void getStoredChatThread(run.threadId).then(thread => {
      if (!cancelled) setProjectId(thread?.projectId ?? null);
    }).catch(() => {});
    return () => { cancelled = true; active.current = false; };
  }, [run.threadId]);

  const remaining = sources.filter(s => !queued.has(s.url));
  const count = remaining.filter(s => selected.has(s.url)).length;
  async function importSelected() {
    if (!projectId || running.current || !count) return;
    running.current = true;
    setBusy(true);
    const target = projectId;
    const policy = useChatRuntimeStore.getState().researchWebsitePolicy;
    try {
      // Sequential network imports; native ingestion batches and bounds its own work.
      for (const source of remaining) {
        if (!active.current) break;
        if (!selected.has(source.url)) continue;
        invalidateProjectSources(target);
        try {
          const result = await importResearchSource(target, run.id, source.url, policy);
          if (active.current) {
            setQueued(prev => new Set(prev).add(source.url));
            setErrors(prev => { const next = { ...prev }; delete next[source.url]; return next; });
          }
          if (result.existing || !result.jobId) {
            if (active.current) setExisting(prev => new Set(prev).add(source.url));
            continue;
          }
          void watchIngestion(target, result.jobId, result.filename || source.title).then(job => {
            if (!active.current || !job) return;
            if (job.status === "completed") {
              setIndexed(prev => new Set(prev).add(source.url));
            } else {
              setQueued(prev => { const next = new Set(prev); next.delete(source.url); return next; });
              setErrors(prev => ({ ...prev, [source.url]: job.error || "Indexing did not complete. Retry this source." }));
            }
          });
        } catch (error) {
          if (active.current) setErrors(prev => ({ ...prev, [source.url]: error instanceof Error ? error.message : String(error) }));
        } finally {
          announceProjectSourcesUpdated(target);
        }
      }
    } finally {
      running.current = false;
      if (active.current) setBusy(false);
    }
  }
  if (!projectId || !sources.length) return null;
  return (
    <div className="my-3 rounded-xl border border-border p-3">
      <Button type="button" variant="outline" size="sm" onClick={() => setExpanded(v => !v)}>
        {expanded ? "Hide source selection" : "Add research sources to project"}
      </Button>
      {expanded && <div className="mt-3 space-y-3">
        <p className="text-xs text-muted-foreground">
          Fetch original accessible source content, not the AI report. Direct PDFs and supported arXiv links are saved as PDF files; other web pages are saved as text snapshots. Added sources use the project's normal embedding and retrieval settings.
        </p>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={remaining.length > 0 && count === remaining.length} disabled={busy || !remaining.length}
            onChange={e => setSelected(e.target.checked ? new Set(remaining.map(s => s.url)) : new Set())} />
          Select all discovered sources
        </label>
        <div className="max-h-80 space-y-2 overflow-y-auto">
          {sources.map(source => <label key={source.url} className="flex items-start gap-2 rounded-lg bg-muted/30 p-2">
            <input type="checkbox" className="mt-1" checked={queued.has(source.url) || selected.has(source.url)}
              disabled={busy || queued.has(source.url)} onChange={e => {
                const checked = e.target.checked;
                setSelected(prev => { const next = new Set(prev); if (checked) next.add(source.url); else next.delete(source.url); return next; });
              }} />
            <span className="min-w-0 flex-1 text-sm">
              <span className="block break-words">{source.title || source.url}</span>
              <span className="block break-all text-xs text-muted-foreground">{source.url}</span>
              {queued.has(source.url) && <span className="block text-xs text-muted-foreground">
                {existing.has(source.url) ? "Already retained in this project" : indexed.has(source.url) ? "Indexed in project" : "Submitted to project indexing"}
              </span>}
              {errors[source.url] && <span role="alert" className="block text-xs text-destructive">{errors[source.url]}</span>}
            </span>
          </label>)}
        </div>
        <Button type="button" size="sm" disabled={busy || count === 0} onClick={() => void importSelected()}>
          {busy ? "Importing sources…" : `Import selected (${count})`}
        </Button>
      </div>}
    </div>
  );
}
