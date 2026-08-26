export type InputMode = "single" | "cross-modal" | "bi-temporal";

export type Modality = "optical" | "sar" | "multispectral";

export type TaskKind =
  | "vqa"
  | "caption"
  | "grounding"
  | "change"
  | "change-vqa"
  | "cross-modal"
  | "measure";

export type Asset = {
  id: string;
  role: "primary" | "secondary";
  modality: Modality;
  name: string;
  src: string;
  sensor: string;
  date: string;
  gsd: string;
  location: string;
  coords: string;
  format: string;
  crs?: string;
};

export type Box = {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  kind?: string;
};

export type Mask = {
  id: string;
  label: string;
  color: string;
  opacity: number;
  d: string;
};

export type LayerId =
  | "flood"
  | "water"
  | "urban"
  | "vegetation"
  | "change"
  | "grounding"
  | "settlements";

export type Layer = {
  id: LayerId;
  label: string;
  color: string;
  active: boolean;
};

export type Metric = {
  label: string;
  value: string;
  hint?: string;
};

export type ModelUse = {
  name: string;
  role: string;
  status: "queued" | "running" | "complete" | "standby";
};

export type AgentStep = {
  id: string;
  label: string;
  detail: string;
  status: "pending" | "running" | "done";
};

export type QueryResult = {
  task: TaskKind;
  title: string;
  answer: string;
  observations: string[];
  metrics: Metric[];
  confidence: number;
  boxes: Box[];
  masks: Mask[];
  layers: Layer[];
  models: ModelUse[];
  trace: AgentStep[];
  compareDefault?: CompareMode;
};

export type CompareMode = "primary" | "secondary" | "split" | "swipe" | "diff";

export type MissionReply = {
  match: string[];
  result: QueryResult;
};

export type Mission = {
  id: string;
  code: string;
  title: string;
  location: string;
  synopsis: string;
  demonstrates: string;
  mode: InputMode;
  assets: Asset[];
  suggested: string[];
  replies: MissionReply[];
  fallback: QueryResult;
};

export type ThreadItem = {
  id: string;
  role: "user" | "instrument";
  text: string;
  result?: QueryResult;
};

export type Intent = {
  task: TaskKind;
  label: string;
  specialists: string[];
  warning?: string;
};
