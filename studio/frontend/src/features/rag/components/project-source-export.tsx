// SPDX-License-Identifier: AGPL-3.0-only
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { useEffect, useRef, useState } from "react";
import { getProjectDocumentExportUrl } from "../api/rag-api";
import type { RagDocument } from "../types/rag";

interface ExportEntry {
  documentId: string;
  filename: string;
  exportedFilename?: string;
  representation?: string;
  sourceUrl?: string | null;
  error?: string;
  extraction?: string | null;
  extractionWarning?: string | null;

export function ProjectSourceExport({ projectId, documents }: {
  projectId: string;
  documents: RagDocument[];
}) {
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [openFolder, setOpenFolder] = useState(true);
  const [progress, setProgress] = useState("");
  const [results, setResults] = useState<ExportEntry[]>([]);
  const active = useRef(false);
  const cancelled = useRef(false);
  const running = useRef(false);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; cancelled.current = true; };
  }, []);
  const available = documents.filter(doc => !doc.id.startsWith("pending_"));
  const chosen = available.filter(doc => selected.has(doc.id));
  async function exportFiles() {
    if (running.current || !chosen.length) return;
    running.current = true;
    cancelled.current = false;
    setBusy(true);
    setResults([]);
    setProgress("Choose a destination folder…");
    const entries: ExportEntry[] = [];
    let token: string | null = null;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      token = await invoke<string | null>("pick_native_download_directory");
      if (!token || cancelled.current) {
        if (active.current) setProgress("Export cancelled.");
        return;
      }
      for (const doc of chosen) {
        if (cancelled.current) break;
        if (active.current) setProgress(`Exporting ${entries.length + 1} of ${chosen.length}: ${doc.filename}`);
        const entry: ExportEntry = { documentId: doc.id, filename: doc.filename, extraction: doc.extraction, extractionWarning: doc.extractionWarning };
        try {
          const source = await getProjectDocumentExportUrl(projectId, doc.id);
          if (cancelled.current) break;
          entry.representation = source.representation;
          entry.sourceUrl = source.sourceUrl;
          entry.exportedFilename = await invoke<string>("save_native_file_in_directory", {
            token, url: source.url, fileName: source.filename.split("/").pop() || source.filename,
          });
        } catch (error) {
          entry.error = error instanceof Error ? error.message : String(error);
        }
        entries.push(entry);
        if (active.current) setResults([...entries]);
      }
      const saved = entries.filter(entry => entry.exportedFilename).length;
      const failed = entries.filter(entry => entry.error).length;
      if (entries.length) {
        try {
          await invoke<string>("save_native_source_manifest", {
            token, manifest: JSON.stringify({
              projectId, cancelled: cancelled.current, requested: chosen.length,
              exported: saved, failed, entries,
            }, null, 2),
          });
        } catch (error) {
          toast.error("Could not save the source manifest", { description: String(error) });
        }
      }
      if (active.current) setProgress(`${cancelled.current ? "Stopped. " : ""}Exported ${saved} of ${chosen.length} sources; ${failed} failed.`);
      if (openFolder && saved > 0 && !cancelled.current && active.current) {
        try { await invoke("open_native_export_directory", { token }); }
        catch (error) { toast.error("Could not open export folder", {description: String(error)}); }
      }
    } catch (error) {
      if (active.current) setProgress(String(error));
    } finally {
      if (token) {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("release_native_download_directory", { token });
        } catch { /* expired directory handle needs no further cleanup */ }
      }
      running.current = false;
      if (active.current) setBusy(false);
    }
  }
  return (
    <div className="mb-4 rounded-2xl border border-border p-4">
      <Button type="button" variant="outline" size="sm" disabled={!available.length || busy}
        onClick={() => { setExpanded(!expanded); if (!expanded) setSelected(new Set(available.map(doc => doc.id))); }}>
        {expanded ? "Hide export options" : "Export sources"}
      </Button>
      {expanded && <div className="mt-3 space-y-3">
        <p className="text-xs text-muted-foreground">Export stored files to a folder. Older web sources may be text snapshots rather than original documents.</p>
        <div className="flex flex-wrap gap-2">
          <Button type="button" size="sm" variant="ghost" disabled={busy} onClick={() => setSelected(new Set(available.map(doc => doc.id)))}>Select all</Button>
          <Button type="button" size="sm" variant="ghost" disabled={busy} onClick={() => setSelected(new Set())}>Deselect all</Button>
        </div>
        <div className="max-h-64 space-y-1 overflow-y-auto">
          {available.map(doc => <label key={doc.id} className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={selected.has(doc.id)} disabled={busy}
              onChange={event => { const checked = event.target.checked; setSelected(prev => {
                const next = new Set(prev); if (checked) next.add(doc.id); else next.delete(doc.id); return next;
              }); }} />
            <span className="break-all">{doc.filename}</span>
          </label>)}
        </div>
        <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={openFolder} disabled={busy} onChange={event => setOpenFolder(event.target.checked)} />Open folder after export</label>
        <div className="flex flex-wrap gap-2">
          <Button type="button" size="sm" disabled={busy || !chosen.length} onClick={() => void exportFiles()}>Export selected ({chosen.length})</Button>
          {busy && <Button type="button" variant="outline" size="sm" onClick={() => { cancelled.current = true; setProgress("Stopping after current file…"); }}>Stop export</Button>}
          {!busy && results.some(row => row.error) && <Button type="button" variant="ghost" size="sm" onClick={() => setSelected(new Set(results.filter(row => row.error).map(row => row.documentId)))}>Select failed files</Button>}
        </div>
        {progress && <p role="status" className="text-sm">{progress}</p>}
        {results.filter(row => row.error).map(row => <p key={row.documentId} className="text-xs text-destructive">{row.filename}: {row.error}</p>)}
      </div>}
    </div>
  );
}
