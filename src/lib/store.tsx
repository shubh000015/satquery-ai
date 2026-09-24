"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AGENT_TIMING, classifyQuery, demoChatReply, matchDemoMission, resolveResult } from "./agent";
import { backendEnabled, reportUrl, streamQuery, uploadAssets } from "./api";
import {
  MAX_UPLOAD_FILES,
  draftsFromPlan,
  planScenes,
  type BandPlan,
  type UploadKind,
} from "./bandPlan";
import { customMission, missions } from "./missions";
import type {
  AgentStep,
  CompareMode,
  InputMode,
  Intent,
  LayerId,
  Mission,
  QueryResult,
  ThreadItem,
} from "./types";

type Screen = "ingress" | "workspace";

type Point = { x: number; y: number };

export type ChatSession = {
  id: string;
  title: string;
  date: number;
};

type Store = {
  screen: Screen;
  mission: Mission | null;
  acquiring: boolean;
  query: string;
  intent: Intent | null;
  running: boolean;
  steps: AgentStep[];
  result: QueryResult | null;
  thread: ThreadItem[];
  history: ChatSession[];
  activeSessionId: string | null;
  compare: CompareMode;
  swipe: number;
  scale: number;
  pan: Point;
  selectedId: string | null;
  reportOpen: boolean;
  auditOpen: boolean;
  measuring: boolean;
  measurePts: Point[];
  pairChoice: boolean;
  pendingFiles: { name: string; src: string }[] | null;
  kindDialog: boolean;
  uploadKind: UploadKind | null;
  bandPlan: BandPlan | null;
  setQuery: (q: string) => void;
  openMission: (id: string, opts?: { autorun?: boolean; query?: string }) => void;
  ingestFiles: (files: FileList | File[], kind?: UploadKind | null) => void;
  openKindDialog: () => void;
  chooseUploadKind: (kind: UploadKind) => void;
  cancelKindDialog: () => void;
  confirmPair: (mode: InputMode) => void;
  cancelPair: () => void;
  submit: (text?: string) => void;
  toggleLayer: (id: LayerId) => void;
  setCompare: (c: CompareMode) => void;
  setSwipe: (n: number) => void;
  setScale: (n: number | ((s: number) => number)) => void;
  setPan: (p: Point | ((p: Point) => Point)) => void;
  resetView: () => void;
  selectFeature: (id: string | null) => void;
  setReportOpen: (v: boolean) => void;
  setAuditOpen: (v: boolean) => void;
  setMeasuring: (v: boolean) => void;
  addMeasurePt: (p: Point) => void;
  goIngress: () => void;
  startNewChat: () => void;
  deleteSession: (id: string) => void;
  removeAsset: (assetId: string) => void;
  setScreen: (s: Screen) => void;
  loadSession: (id: string) => void;
  /** True when NEXT_PUBLIC_SATQUERY_API is set and uploads are analysed server-side. */
  backendLive: boolean;
  /** Markdown report for the current run, or null on the scripted missions. */
  reportHref: string | null;
};

const Ctx = createContext<Store | null>(null);

async function filesFromMission(mission: Mission): Promise<File[]> {
  const files: File[] = [];
  for (const asset of mission.assets) {
    if (!asset.src || asset.src.startsWith("blob:")) continue;
    const response = await fetch(asset.src);
    if (!response.ok) throw new Error(`Could not load ${asset.name || asset.src}`);
    const blob = await response.blob();
    const name = asset.src.split("/").pop() || `${asset.id}.jpg`;
    files.push(new File([blob], name, { type: blob.type || "image/jpeg" }));
  }
  return files;
}

export function SatQueryProvider({ children }: { children: ReactNode }) {
  const [screen, setScreen] = useState<Screen>("ingress");
  const [mission, setMission] = useState<Mission | null>(null);
  const [acquiring, setAcquiring] = useState(false);
  const [query, setQueryState] = useState("");
  const [intent, setIntent] = useState<Intent | null>(null);
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [thread, setThread] = useState<ThreadItem[]>([]);
  const [history, setHistory] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [compare, setCompare] = useState<CompareMode>("primary");
  const [swipe, setSwipe] = useState(52);
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [reportOpen, setReportOpen] = useState(false);
  const [auditOpen, setAuditOpen] = useState(false);
  const [measuring, setMeasuring] = useState(false);
  const [measurePts, setMeasurePts] = useState<Point[]>([]);
  const [pairChoice, setPairChoice] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<{ name: string; src: string }[] | null>(null);
  const [kindDialog, setKindDialog] = useState(false);
  const [uploadKind, setUploadKind] = useState<UploadKind | null>(null);
  const [bandPlan, setBandPlan] = useState<BandPlan | null>(null);
  const previewUrls = useRef<string[]>([]);
  const runId = useRef(0);
  const urlBooted = useRef(false);
  const autoRan = useRef(false);
  const pendingAsk = useRef<string | null>(null);
  // Uploaded File handles are kept so they can be posted to the agent backend on
  // the first query; the blob URLs above are only good for display.
  const uploadedFiles = useRef<File[]>([]);
  const remote = useRef<{ sessionId: string; assetIds: string[]; signature: string } | null>(null);
  const [reportHref, setReportHref] = useState<string | null>(null);

  const setQuery = useCallback((q: string) => {
    setQueryState(q);
    if (!mission) {
      setIntent(null);
      return;
    }
    setIntent(q.trim() ? classifyQuery(q, mission.mode) : null);
  }, [mission]);

  const bootMission = useCallback((m: Mission) => {
    setMission(m);
    setResult(null);
    setThread([]);
    setSteps([]);
    setQueryState("");
    setIntent(null);
    setSelectedId(null);
    setAuditOpen(false);
    setReportOpen(false);
    setMeasuring(false);
    setMeasurePts([]);
    setScale(1);
    setPan({ x: 0, y: 0 });
    setSwipe(52);
    setCompare(m.mode === "single" ? "primary" : "split");
    setAcquiring(true);
    setScreen("workspace");
    window.setTimeout(() => setAcquiring(false), 1400);
  }, []);

  const openMission = useCallback((id: string, opts?: { autorun?: boolean; query?: string }) => {
    const m = missions.find((x) => x.id === id);
    if (!m) return;
    const asked = opts?.query?.trim();
    pendingAsk.current = asked || null;
    bootMission(m);
  }, [bootMission]);

  useEffect(() => {
    if (urlBooted.current) return;
    const id = new URLSearchParams(window.location.search).get("mission");
    if (!id) return;
    urlBooted.current = true;
    openMission(id);
  }, [openMission]);

  const goIngress = useCallback(() => {
    runId.current += 1;
    pendingAsk.current = null;
    autoRan.current = false;
    uploadedFiles.current = [];
    remote.current = null;
    previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
    previewUrls.current = [];
    setBandPlan(null);
    setUploadKind(null);
    setKindDialog(false);
    setReportHref(null);
    setScreen("ingress");
    setMission(null);
    setRunning(false);
    setAcquiring(false);
    setActiveSessionId(null);
  }, []);

  const startNewChat = useCallback(() => {
    runId.current += 1;
    pendingAsk.current = null;
    autoRan.current = false;
    uploadedFiles.current = [];
    remote.current = null;
    previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
    previewUrls.current = [];
    setBandPlan(null);
    setUploadKind(null);
    setKindDialog(false);
    setReportHref(null);
    setMission(null);
    setResult(null);
    setThread([]);
    setSteps([]);
    setRunning(false);
    setAcquiring(false);
    setActiveSessionId(null);
    setScreen("workspace");
  }, []);

  const deleteSession = useCallback((id: string) => {
    setHistory((prev) => prev.filter((h) => h.id !== id));
    if (activeSessionId === id) {
      startNewChat();
    }
  }, [activeSessionId, startNewChat]);

    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    const applyFileSet = useCallback((files: File[], kind: UploadKind | null) => {
      const plan = planScenes(files.map((f) => f.name), kind);
      setBandPlan(plan);
      if (plan.needsKind && !kind) {
        setKindDialog(true);
        return;
      }

      const drafts = draftsFromPlan(plan, files);
      previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
      previewUrls.current = drafts.map((d) => d.src).filter(Boolean);

      const sceneCount = drafts.length;
      if (sceneCount === 2 && plan.ready && screen === "ingress") {
        setPendingFiles(drafts);
        setPairChoice(true);
        return;
      }

      const m = customMission(drafts);
      const isCustom = mission?.id === "upload";
      if (isCustom) {
        setMission(m);
        setCompare(m.mode === "single" ? "primary" : "split");
        setAcquiring(true);
        window.setTimeout(() => setAcquiring(false), 1400);
      } else {
        bootMission(m);
      }
    }, [bootMission, mission?.id, screen]);

    const ingestFiles = useCallback((list: FileList | File[], kind?: UploadKind | null) => {
      const incoming = Array.from(list).filter((f) =>
        /\.(tif|tiff|png|jpe?g|webp)$/i.test(f.name) || f.type.startsWith("image/")
      );
      const chosen = kind ?? uploadKind;
      const isCustom = mission?.id === "upload";
      const existing = isCustom || incoming.length === 0 ? uploadedFiles.current : [];
      const merged: File[] = [...existing];
      for (const file of incoming) {
        if (merged.some((f) => f.name === file.name && f.size === file.size)) continue;
        merged.push(file);
      }
      if (!merged.length) return;
      if (merged.length > MAX_UPLOAD_FILES) {
        setErrorMsg(
          `At most ${MAX_UPLOAD_FILES} files — 12 optical bands, 2 SAR bands, or both.`
        );
        window.setTimeout(() => setErrorMsg(null), 4000);
        return;
      }

      uploadedFiles.current = merged;
      remote.current = null;
      setReportHref(null);
      applyFileSet(merged, chosen);
    }, [applyFileSet, mission?.id, uploadKind]);

    const openKindDialog = useCallback(() => setKindDialog(true), []);

    const chooseUploadKind = useCallback((kind: UploadKind) => {
      setUploadKind(kind);
      setKindDialog(false);
      if (uploadedFiles.current.length) applyFileSet(uploadedFiles.current, kind);
    }, [applyFileSet]);

    const cancelKindDialog = useCallback(() => setKindDialog(false), []);

  const removeAsset = useCallback((assetId: string) => {
    if (mission?.id !== "upload") return;
    const removed = mission.assets.find((a) => a.id === assetId);
    const drop = new Set(removed?.sourceFiles?.length ? removed.sourceFiles : [removed?.name ?? ""]);
    uploadedFiles.current = uploadedFiles.current.filter((f) => !drop.has(f.name));
    remote.current = null;
    setReportHref(null);
    if (uploadedFiles.current.length === 0) {
      setMission(null);
      setResult(null);
      setBandPlan(null);
      return;
    }
    applyFileSet(uploadedFiles.current, uploadKind);
  }, [applyFileSet, mission, uploadKind]);

  const confirmPair = useCallback((mode: InputMode) => {
    if (!pendingFiles) return;
    const m = customMission(pendingFiles);
    m.mode = mode === "single" ? "cross-modal" : mode;
    if (mode === "bi-temporal") {
      m.assets = m.assets.map((a, i) => ({
        ...a,
        modality: "optical",
        name: i === 0 ? `${a.name} · t₀` : `${a.name} · t₁`,
      }));
    }
    setPairChoice(false);
    setPendingFiles(null);
    remote.current = null; // the declared pair mode is a hint the backend needs
    bootMission(m);
  }, [pendingFiles, bootMission]);

  const cancelPair = useCallback(() => {
    pendingFiles?.forEach((f) => URL.revokeObjectURL(f.src));
    setPendingFiles(null);
    setPairChoice(false);
  }, [pendingFiles]);

  const loadSession = useCallback((id: string) => {
    setActiveSessionId(id);
    setScreen("workspace");
    // Mock loading logic, in reality we'd fetch full session details
  }, []);

  /** Scripted mission path: animate the canned trace, then reveal the result. */
  const localRun = useCallback((q: string, m: Mission, id: number) => {
    const next = resolveResult(q, m);
    setSteps(next.trace.map((s) => ({ ...s, status: "pending" })));

    let i = 0;
    const tick = () => {
      if (runId.current !== id) return;
      setSteps((prev) =>
        prev.map((s, idx) => {
          if (idx < i) return { ...s, status: "done" };
          if (idx === i) return { ...s, status: "running" };
          return s;
        })
      );
      i += 1;
      if (i < next.trace.length) {
        window.setTimeout(tick, AGENT_TIMING[i] ?? 600);
      } else {
        window.setTimeout(() => {
          if (runId.current !== id) return;
          setSteps((prev) => prev.map((s) => ({ ...s, status: "done" })));
          setResult(next);
          if (next.compareDefault) setCompare(next.compareDefault);
          setRunning(false);
          setThread((t) => [
            ...t,
            { id: `a-${id}`, role: "instrument", text: next.answer, result: next },
          ]);
          window.setTimeout(() => {
            if (runId.current === id) setAuditOpen(false);
          }, 2400);
        }, 480);
      }
    };
    window.setTimeout(tick, AGENT_TIMING[0]);
  }, []);

  const flashError = useCallback((message: string) => {
    setErrorMsg(message);
    window.setTimeout(() => setErrorMsg(null), 4000);
  }, []);

  /**
   * Agent-backend path for uploaded scenes: post the files once per input set,
   * then stream the pipeline so the audit trail fills in stage by stage.
   */
  const remoteRun = useCallback(async (q: string, m: Mission, id: number) => {
    try {
      const signature = uploadedFiles.current
        .map((f) => `${f.name}:${f.size}:${f.lastModified}`)
        .join("|");

      let session = remote.current;
      if (!session || session.signature !== signature) {
        const upload = await uploadAssets(uploadedFiles.current, {
          mode: m.mode,
          expect: uploadKind ?? undefined,
        });
        session = {
          sessionId: upload.sessionId,
          assetIds: upload.assets.map((a) => a.id),
          signature,
        };
        remote.current = session;
        const blocking = upload.validation.issues.filter((i) => i.severity === "error");
        if (blocking.length) flashError(blocking[0].message);
      }

      const result = await streamQuery(
        { query: q, assetIds: session.assetIds, sessionId: session.sessionId, mode: m.mode },
        {
          onTrace: (steps) => {
            if (runId.current === id) setSteps(steps);
          },
          onStep: (step) => {
            if (runId.current !== id) return;
            setSteps((prev) => prev.map((s) => (s.id === step.id ? step : s)));
          },
        }
      );

      if (runId.current !== id) return;
      setResult(result);
      if (result.compareDefault) setCompare(result.compareDefault);
      setRunning(false);
      setReportHref(result.sessionId ? reportUrl(result.sessionId, result.queryId) : null);
      setThread((t) => [
        ...t,
        { id: `a-${id}`, role: "instrument", text: result.answer, result },
      ]);
      window.setTimeout(() => {
        if (runId.current === id) setAuditOpen(false);
      }, 2400);
    } catch {
      if (runId.current !== id) return;
      localRun(q, m, id);
    }
  }, [flashError, localRun, uploadKind]);

  const rememberSession = useCallback((q: string) => {
    let currentSessionId = activeSessionId;
    if (!currentSessionId) {
      currentSessionId = String(runId.current + 1);
      setActiveSessionId(currentSessionId);
      setHistory((prev) => {
        if (prev.some((h) => h.id === currentSessionId)) return prev;
        const words = q.split(" ");
        const title = words.slice(0, 4).join(" ") + (words.length > 4 ? "..." : "");
        return [{ id: currentSessionId!, title, date: Date.now() }, ...prev];
      });
    }
  }, [activeSessionId]);

  const beginRun = useCallback((q: string, m: Mission) => {
    const classified = classifyQuery(q, m.mode);
    setIntent(classified);
    setQueryState("");
    rememberSession(q);

    const id = ++runId.current;
    setRunning(true);
    setAuditOpen(true);
    setSelectedId(null);
    setMeasuring(false);
    setResult(null);
    setThread((t) => [...t, { id: `u-${id}`, role: "user", text: q }]);

    if (backendEnabled()) {
      setSteps([]);
      void (async () => {
        try {
          if (uploadedFiles.current.length === 0 && m.assets.length > 0) {
            uploadedFiles.current = await filesFromMission(m);
            remote.current = null;
          }
          if (uploadedFiles.current.length === 0) {
            flashError("Upload a satellite scene first.");
            setRunning(false);
            return;
          }
          await remoteRun(q, m, id);
        } catch (err) {
          if (runId.current !== id) return;
          const detail = err instanceof Error ? err.message : "Could not send the scene to the backend.";
          flashError(detail);
          setRunning(false);
        }
      })();
      return;
    }
    localRun(q, m, id);
  }, [flashError, localRun, remoteRun, rememberSession]);

  const submit = useCallback((text?: string) => {
    if (running) return;
    const q = (text ?? query).trim();
    if (!q) return;

    // No scene loaded: do not map the question onto a canned demo. Ask for
    // an upload so the backend can analyse a real file.
    if (!mission) {
      if (!backendEnabled()) {
        const demo = missions.find((m) => m.id === matchDemoMission(q));
        if (demo) {
          setMission(demo);
          setResult(null);
          setSteps([]);
          setSelectedId(null);
          setReportOpen(false);
          setMeasuring(false);
          setMeasurePts([]);
          setScale(1);
          setPan({ x: 0, y: 0 });
          setSwipe(52);
          setCompare(demo.mode === "single" ? "primary" : "split");
          setAcquiring(false);
          setReportHref(null);
          setScreen("workspace");
          beginRun(q, demo);
          return;
        }
      }

      rememberSession(q);
      const id = ++runId.current;
      setScreen("workspace");
      setQueryState("");
      setIntent(null);
      setRunning(true);
      setThread((t) => [...t, { id: `u-${id}`, role: "user", text: q }]);
      window.setTimeout(() => {
        if (runId.current !== id) return;
        setRunning(false);
        setThread((t) => [
          ...t,
          { id: `a-${id}`, role: "instrument", text: demoChatReply(q) },
        ]);
      }, 420);
      return;
    }

    beginRun(q, mission);
  }, [mission, query, running, beginRun, rememberSession]);

  useEffect(() => {
    if (!mission || acquiring || running) return;
    if (pendingAsk.current) {
      const text = pendingAsk.current;
      pendingAsk.current = null;
      window.setTimeout(() => submit(text), 280);
      return;
    }
    if (autoRan.current) return;
    const params = new URLSearchParams(window.location.search);
    const raw = params.get("q");
    if (!raw) return;
    autoRan.current = true;
    window.setTimeout(() => submit(raw), 240);
  }, [mission, acquiring, running, submit]);

  const toggleLayer = useCallback((id: LayerId) => {
    setResult((r) =>
      r
        ? { ...r, layers: r.layers.map((l) => (l.id === id ? { ...l, active: !l.active } : l)) }
        : r
    );
  }, []);

  const addMeasurePt = useCallback((p: Point) => {
    setMeasurePts((pts) => (pts.length >= 2 ? [p] : [...pts, p]));
  }, []);



  const value = useMemo<Store>(
    () => ({
      screen,
      mission,
      acquiring,
      query,
      intent,
      running,
      steps,
      result,
      thread,
      history,
      activeSessionId,
      compare,
      swipe,
      scale,
      pan,
      selectedId,
      reportOpen,
      auditOpen,
      measuring,
      measurePts,
      pairChoice,
      pendingFiles,
      kindDialog,
      uploadKind,
      bandPlan,
      setQuery,
      openMission,
      ingestFiles,
      openKindDialog,
      chooseUploadKind,
      cancelKindDialog,
      confirmPair,
      cancelPair,
      submit,
      toggleLayer,
      setCompare,
      setSwipe,
      setScale,
      setPan,
      resetView: () => {
        setScale(1);
        setPan({ x: 0, y: 0 });
      },
      selectFeature: setSelectedId,
      setReportOpen,
      setAuditOpen,
      setMeasuring: (v) => {
        setMeasuring(v);
        if (!v) setMeasurePts([]);
      },
      addMeasurePt,
      goIngress,
      startNewChat,
      deleteSession,
      removeAsset,
      setScreen,
      loadSession,
      backendLive: backendEnabled(),
      reportHref,
    }),
    [
      screen,
      mission,
      acquiring,
      query,
      intent,
      running,
      steps,
      result,
      thread,
      history,
      activeSessionId,
      compare,
      swipe,
      scale,
      pan,
      selectedId,
      reportOpen,
      auditOpen,
      measuring,
      measurePts,
      pairChoice,
      pendingFiles,
      kindDialog,
      uploadKind,
      bandPlan,
      setQuery,
      openMission,
      ingestFiles,
      openKindDialog,
      chooseUploadKind,
      cancelKindDialog,
      confirmPair,
      cancelPair,
      submit,
      toggleLayer,
      addMeasurePt,
      goIngress,
      startNewChat,
      deleteSession,
      removeAsset,
      setScreen,
      loadSession,
      reportHref,
    ]
  );

  return (
    <Ctx.Provider value={value}>
      {children}
      {errorMsg && (
        <div className="fixed top-6 left-1/2 -translate-x-1/2 z-[9999] rounded-lg bg-red-500/90 px-4 py-2 text-sm font-medium text-white shadow-lg backdrop-blur-md transition-all">
          {errorMsg}
        </div>
      )}
    </Ctx.Provider>
  );
}

export function useSatQuery() {
  const v = useContext(Ctx);
  if (!v) throw new Error("SatQuery store missing");
  return v;
}


