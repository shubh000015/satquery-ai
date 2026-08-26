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
import { AGENT_TIMING, classifyQuery, resolveResult } from "./agent";
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
  setQuery: (q: string) => void;
  openMission: (id: string, opts?: { autorun?: boolean; query?: string }) => void;
  ingestFiles: (files: FileList | File[]) => void;
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
};

const Ctx = createContext<Store | null>(null);

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
  const runId = useRef(0);
  const urlBooted = useRef(false);
  const autoRan = useRef(false);
  const pendingAsk = useRef<string | null>(null);

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
    setCompare(m.mode === "single" ? "primary" : "swipe");
    setAcquiring(true);
    setScreen("workspace");
    window.setTimeout(() => setAcquiring(false), 1400);
  }, []);

  const openMission = useCallback((id: string, opts?: { autorun?: boolean; query?: string }) => {
    const m = missions.find((x) => x.id === id);
    if (!m) return;
    const asked = opts?.query?.trim();
    pendingAsk.current = asked || (opts?.autorun ? m.suggested[0] : null);
    bootMission(m);
  }, [bootMission]);

  useEffect(() => {
    if (urlBooted.current) return;
    const id = new URLSearchParams(window.location.search).get("mission");
    if (!id) return;
    urlBooted.current = true;
    openMission(id);
  }, [openMission]);

  const ingestFiles = useCallback((list: FileList | File[]) => {
    const files = Array.from(list).filter((f) =>
      /\.(tif|tiff|png|jpe?g|webp)$/i.test(f.name) || f.type.startsWith("image/")
    );
    if (!files.length) return;
    const mapped = files.slice(0, 2).map((f) => ({ name: f.name, src: URL.createObjectURL(f) }));
    if (mapped.length >= 2) {
      setPendingFiles(mapped);
      setPairChoice(true);
      return;
    }
    bootMission(customMission(mapped));
  }, [bootMission]);

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
    bootMission(m);
  }, [pendingFiles, bootMission]);

  const cancelPair = useCallback(() => {
    pendingFiles?.forEach((f) => URL.revokeObjectURL(f.src));
    setPendingFiles(null);
    setPairChoice(false);
  }, [pendingFiles]);

  const submit = useCallback((text?: string) => {
    if (!mission || running) return;
    const q = (text ?? query).trim();
    if (!q) return;
    const classified = classifyQuery(q, mission.mode);
    setIntent(classified);
    if (text) setQueryState(text);

    const next = resolveResult(q, mission);
    const id = ++runId.current;
    setRunning(true);
    setAuditOpen(true);
    setSelectedId(null);
    setMeasuring(false);
    setResult(null);
    setSteps(next.trace.map((s) => ({ ...s, status: "pending" })));
    setThread((t) => [...t, { id: `u-${id}`, role: "user", text: q }]);

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
  }, [mission, query, running]);

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
    const auto = params.has("autorun");
    if (!raw && !auto) return;
    autoRan.current = true;
    const text = raw || mission.suggested[0];
    window.setTimeout(() => submit(text), 240);
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

  const goIngress = useCallback(() => {
    runId.current += 1;
    pendingAsk.current = null;
    autoRan.current = false;
    setScreen("ingress");
    setMission(null);
    setRunning(false);
    setAcquiring(false);
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
      setQuery,
      openMission,
      ingestFiles,
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
      setQuery,
      openMission,
      ingestFiles,
      confirmPair,
      cancelPair,
      submit,
      toggleLayer,
      addMeasurePt,
      goIngress,
    ]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSatQuery() {
  const v = useContext(Ctx);
  if (!v) throw new Error("SatQuery store missing");
  return v;
}
