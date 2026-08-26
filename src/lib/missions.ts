import type { Mission, QueryResult } from "./types";

const kosiFlood: QueryResult = {
  task: "cross-modal",
  title: "Inundation on the Kosi fan",
  answer:
    "Standing water is confirmed across the left-bank floodplain. Optical colour is ambiguous under thin cloud, but SAR backscatter collapses over specular water, so the fused mask is reliable. Seven settlement clusters intersect the inundation at 10 m GSD.",
  observations: [
    "Primary inundation follows the braided Kosi channel and spills east into standing-crop parcels.",
    "Seven village clusters sit on the flood edge; three are partially surrounded.",
    "Optical water index alone under-calls extent in the cloud-shadowed north-west quadrant.",
    "SAR VV returns remain low through that quadrant, recovering 2.1 km² the optical stack missed.",
  ],
  metrics: [
    { label: "Flooded area", value: "41.6 km²", hint: "fused mask" },
    { label: "Permanent water", value: "9.4 km²", hint: "pre-event channel" },
    { label: "Settlements hit", value: "7", hint: "intersect mask" },
    { label: "Cloud-affected", value: "18%", hint: "optical only" },
  ],
  confidence: 0.936,
  boxes: [
    { id: "s1", label: "Settlement A", x: 34, y: 22, w: 7, h: 6, score: 0.91, kind: "settlement" },
    { id: "s2", label: "Settlement B", x: 46, y: 31, w: 6, h: 5, score: 0.88, kind: "settlement" },
    { id: "s3", label: "Settlement C", x: 41, y: 48, w: 8, h: 6, score: 0.93, kind: "settlement" },
    { id: "s4", label: "Settlement D", x: 58, y: 44, w: 6, h: 5, score: 0.84, kind: "settlement" },
    { id: "s5", label: "Settlement E", x: 52, y: 62, w: 7, h: 6, score: 0.9, kind: "settlement" },
    { id: "s6", label: "Settlement F", x: 68, y: 28, w: 6, h: 5, score: 0.81, kind: "settlement" },
    { id: "s7", label: "Settlement G", x: 29, y: 70, w: 7, h: 6, score: 0.87, kind: "settlement" },
  ],
  masks: [
    {
      id: "flood",
      label: "Inundation",
      color: "#4ea3ff",
      opacity: 0.38,
      d: "M0 6 C 10 8, 16 18, 14 32 C 18 44, 12 52, 17 64 C 11 74, 16 84, 9 94 L 0 100 L 0 0 Z M 18 38 C 28 40, 34 46, 30 54 C 24 58, 18 52, 18 38 Z M 8 70 C 20 68, 26 76, 22 84 C 14 86, 8 80, 8 70 Z",
    },
    {
      id: "channel",
      label: "Permanent channel",
      color: "#1d4ed8",
      opacity: 0.45,
      d: "M0 10 C 6 18, 8 40, 7 58 C 6 74, 4 88, 0 96 L 0 10 Z",
    },
  ],
  layers: [
    { id: "flood", label: "Inundation", color: "#4ea3ff", active: true },
    { id: "water", label: "Permanent water", color: "#1d4ed8", active: true },
    { id: "settlements", label: "Settlements", color: "#d4b56a", active: true },
    { id: "vegetation", label: "Standing crop", color: "#7dce9a", active: false },
  ],
  models: [
    { name: "Input Guardian", role: "format · CRS · co-registration", status: "complete" },
    { name: "RSVQA-Adapter", role: "query understanding", status: "complete" },
    { name: "OptSAR Fusion", role: "water / built-up extraction", status: "complete" },
    { name: "RS-Grounder", role: "settlement localisation", status: "complete" },
  ],
  trace: [
    { id: "v", label: "Input validation", detail: "GeoTIFF pair · EPSG:4326 · 10 m · overlap 99.4%", status: "pending" },
    { id: "m", label: "Modality detection", detail: "Primary optical (MSI) · secondary SAR (VV/VH)", status: "pending" },
    { id: "q", label: "Query parse", detail: "Flood extent + affected settlements → fusion + grounding", status: "pending" },
    { id: "t", label: "Tool selection", detail: "OptSAR Fusion → RS-Grounder → Integrator", status: "pending" },
    { id: "e", label: "Visual evidence", detail: "Inundation mask + 7 oriented settlement boxes", status: "pending" },
    { id: "i", label: "Integration", detail: "Text, mask, boxes, confidence fused into one brief", status: "pending" },
  ],
  compareDefault: "swipe",
};

const mundraGround: QueryResult = {
  task: "grounding",
  title: "Harbour objects, Mundra",
  answer:
    "The scene is a deep-water container terminal. Berthed vessels line the quay; gantry cranes sit on the water-facing edge; cylindrical tanks sit inland of the western yard. Grounding boxes below are query-conditioned and scored.",
  observations: [
    "Four large container vessels are alongside the main quay.",
    "One vessel is underway in the approach channel, wake visible.",
    "Tank farm occupies the inner western apron — likely liquid bulk.",
    "Mangrove fringe remains intact on the north-eastern creek.",
  ],
  metrics: [
    { label: "Ships (berthed)", value: "4", hint: "quay" },
    { label: "Ships (underway)", value: "1", hint: "channel" },
    { label: "Tank farm", value: "1 cluster", hint: "west apron" },
    { label: "Cranes", value: "11", hint: "STS line" },
  ],
  confidence: 0.951,
  boxes: [
    { id: "ship1", label: "Vessel 01", x: 14, y: 54, w: 22, h: 7, score: 0.96, kind: "ship" },
    { id: "ship2", label: "Vessel 02", x: 38, y: 50, w: 16, h: 6, score: 0.94, kind: "ship" },
    { id: "ship3", label: "Vessel 03", x: 58, y: 46, w: 14, h: 6, score: 0.92, kind: "ship" },
    { id: "ship4", label: "Vessel 04", x: 72, y: 42, w: 12, h: 5, score: 0.89, kind: "ship" },
    { id: "wake", label: "Underway", x: 6, y: 64, w: 9, h: 5, score: 0.87, kind: "ship" },
    { id: "tanks", label: "Tank farm", x: 16, y: 10, w: 13, h: 12, score: 0.93, kind: "tank" },
  ],
  masks: [],
  layers: [
    { id: "grounding", label: "Grounded objects", color: "#d4b56a", active: true },
    { id: "water", label: "Water", color: "#4ea3ff", active: false },
    { id: "urban", label: "Built-up / yard", color: "#f0c674", active: false },
  ],
  models: [
    { name: "Input Guardian", role: "single-frame optical", status: "complete" },
    { name: "RSVQA-Adapter", role: "object class from query", status: "complete" },
    { name: "RS-Grounder", role: "oriented boxes", status: "complete" },
    { name: "VRS-Captioner", role: "scene context", status: "standby" },
  ],
  trace: [
    { id: "v", label: "Input validation", detail: "Single optical frame · JPEG/benchmark-legal · 0.5 m class GSD", status: "pending" },
    { id: "m", label: "Modality detection", detail: "Optical RGB · no SAR pair attached", status: "pending" },
    { id: "q", label: "Query parse", detail: "Grounding: ships and bulk tanks", status: "pending" },
    { id: "t", label: "Tool selection", detail: "RS-Grounder with VQA class prior", status: "pending" },
    { id: "e", label: "Visual evidence", detail: "6 boxes · mean IoU proxy 0.81 vs. lexicon", status: "pending" },
    { id: "i", label: "Integration", detail: "Counts + boxes + confidence", status: "pending" },
  ],
};

const mundraVqa: QueryResult = {
  ...mundraGround,
  task: "vqa",
  title: "How many ships are in this harbour?",
  answer:
    "Five ships are visible: four berthed along the container quay and one underway in the western approach, leaving a wake. Eleven quay cranes serve the berths.",
  confidence: 0.962,
};

const mundraCaption: QueryResult = {
  task: "caption",
  title: "Scene description",
  answer:
    "High-resolution optical view of Mundra’s deep-water terminal. Two container yards packed in coloured stacks face a sediment-rich channel. Inland: warehouses, a tank farm, and a bulk pad. A mangrove creek bounds the north-east. One ship is steaming out; four remain alongside.",
  observations: [
    "Land cover is industrial / port, not residential.",
    "Water is turbid — tidal sediment, not clear-sea.",
    "Vegetation is confined to the mangrove fringe.",
  ],
  metrics: [
    { label: "Dominant cover", value: "Port / industry" },
    { label: "Water fraction", value: "34%" },
    { label: "Vegetation", value: "8%" },
  ],
  confidence: 0.91,
  boxes: [],
  masks: [],
  layers: [
    { id: "urban", label: "Built-up", color: "#f0c674", active: true },
    { id: "water", label: "Water", color: "#4ea3ff", active: true },
    { id: "vegetation", label: "Mangrove", color: "#7dce9a", active: true },
  ],
  models: [
    { name: "VRS-Captioner", role: "dense caption (VRSBench-style)", status: "complete" },
    { name: "Land-cover Lexicon", role: "BigEarthNet classes", status: "complete" },
  ],
  trace: [
    { id: "v", label: "Input validation", detail: "Single optical frame accepted", status: "pending" },
    { id: "m", label: "Modality detection", detail: "Optical", status: "pending" },
    { id: "q", label: "Query parse", detail: "Caption / land-cover description", status: "pending" },
    { id: "t", label: "Tool selection", detail: "VRS-Captioner", status: "pending" },
    { id: "e", label: "Visual evidence", detail: "Cover classes painted as layers", status: "pending" },
    { id: "i", label: "Integration", detail: "Narrative + class fractions", status: "pending" },
  ],
};

const gurgaonChange: QueryResult = {
  task: "change-vqa",
  title: "Built-up has increased",
  answer:
    "Built-up area has increased. The 1991 Landsat frame is a peri-urban mosaic of fallow plots and village clusters. The contemporary 0.5 m optical frame is a continuous urban fabric — arterial roads, gated campuses, and high-reflectance rooftops. Net built-up gain is estimated at +312% inside this chip.",
  observations: [
    "The western corridor (NH-48 analogue) is the spine of new fabric.",
    "A large green rectangle persists — likely a preserved park / forest patch.",
    "Open sandy parcels in 1991 are almost entirely roofed now.",
    "No evidence of water-body loss at this chip scale; the north-west tank / lake remains.",
  ],
  metrics: [
    { label: "Built-up 1991", value: "18%", hint: "chip fraction" },
    { label: "Built-up now", value: "74%", hint: "chip fraction" },
    { label: "Direction", value: "Increased" },
    { label: "Baseline", value: "33 yr", hint: "1991 → 2024" },
  ],
  confidence: 0.944,
  boxes: [],
  masks: [
    {
      id: "gain",
      label: "Built-up gain",
      color: "#ff6b8a",
      opacity: 0.32,
      d: "M8 12 L 42 10 L 46 48 L 18 62 L 6 40 Z M 48 18 L 92 14 L 94 70 L 52 78 L 46 36 Z M 20 68 L 70 64 L 78 94 L 16 96 Z",
    },
  ],
  layers: [
    { id: "change", label: "Built-up gain", color: "#ff6b8a", active: true },
    { id: "urban", label: "Current fabric", color: "#f0c674", active: false },
    { id: "vegetation", label: "Persistent green", color: "#7dce9a", active: false },
  ],
  models: [
    { name: "Temporal Aligner", role: "chip co-registration", status: "complete" },
    { name: "CD-Mapper", role: "change mask", status: "complete" },
    { name: "CDVQA", role: "increase / decrease / unchanged", status: "complete" },
  ],
  trace: [
    { id: "v", label: "Input validation", detail: "Bi-temporal pair · same footprint · different GSD (resampled)", status: "pending" },
    { id: "m", label: "Modality detection", detail: "Optical t₀ Landsat · optical t₁ VHR", status: "pending" },
    { id: "q", label: "Query parse", detail: "Change VQA: has built-up increased?", status: "pending" },
    { id: "t", label: "Tool selection", detail: "CD-Mapper + CDVQA", status: "pending" },
    { id: "e", label: "Visual evidence", detail: "Gain mask over t₁", status: "pending" },
    { id: "i", label: "Integration", detail: "Ternary answer + fractions + mask", status: "pending" },
  ],
  compareDefault: "swipe",
};

const punjabCaption: QueryResult = {
  task: "caption",
  title: "Doaba land-cover",
  answer:
    "A compact urban core sits in a rabi agricultural mosaic. Fields are rectangular, mixed green and harvested tan, typical of the Punjab doab. Arterials radiate from the settlement. A dark linear feature in the south-east is consistent with a rail / canal alignment. No standing flood water is evident in this chip.",
  observations: [
    "Agriculture is the majority class by area.",
    "Urban fabric is dense but contained — not a megacity sprawl chip.",
    "Field texture supports crop vs. fallow discrimination.",
    "Village satellites punctuate the rural matrix.",
  ],
  metrics: [
    { label: "Agriculture", value: "71%" },
    { label: "Built-up", value: "22%" },
    { label: "Bare / fallow", value: "5%" },
    { label: "Water", value: "2%" },
  ],
  confidence: 0.918,
  boxes: [
    { id: "city", label: "Urban core", x: 28, y: 8, w: 38, h: 28, score: 0.9, kind: "urban" },
  ],
  masks: [
    {
      id: "ag",
      label: "Agriculture",
      color: "#7dce9a",
      opacity: 0.22,
      d: "M0 40 L 100 36 L 100 100 L 0 100 Z",
    },
  ],
  layers: [
    { id: "vegetation", label: "Agriculture", color: "#7dce9a", active: true },
    { id: "urban", label: "Built-up", color: "#f0c674", active: true },
    { id: "water", label: "Water", color: "#4ea3ff", active: false },
  ],
  models: [
    { name: "VRS-Captioner", role: "scene description", status: "complete" },
    { name: "Land-cover Lexicon", role: "BigEarthNet-adapted classes", status: "complete" },
    { name: "RSVQA-Adapter", role: "follow-up VQA", status: "standby" },
  ],
  trace: [
    { id: "v", label: "Input validation", detail: "Single optical · 10 m class", status: "pending" },
    { id: "m", label: "Modality detection", detail: "Optical / MSI", status: "pending" },
    { id: "q", label: "Query parse", detail: "Caption + land-cover inventory", status: "pending" },
    { id: "t", label: "Tool selection", detail: "VRS-Captioner + lexicon", status: "pending" },
    { id: "e", label: "Visual evidence", detail: "Urban box + agriculture wash", status: "pending" },
    { id: "i", label: "Integration", detail: "Narrative + class table", status: "pending" },
  ],
};

const punjabVqa: QueryResult = {
  ...punjabCaption,
  task: "vqa",
  title: "Is agriculture the dominant class?",
  answer:
    "Yes. Agricultural parcels occupy approximately 71% of the chip. Built-up is the second class at 22%, concentrated in the northern urban core. This is a classic urban–rural fringe scene, not a flood or port scene.",
  confidence: 0.94,
};

export const missions: Mission[] = [
  {
    id: "kosi",
    code: "SQ-04",
    title: "Kosi Fan Inundation",
    location: "Saharsa belt, Bihar",
    synopsis: "Cloud-affected optical plus SAR of a braided Himalayan fan. The query that wins this dossier is the one a district officer actually asks.",
    demonstrates: "Optical–SAR fusion · flood VQA · settlement grounding",
    mode: "cross-modal",
    assets: [
      {
        id: "kosi-opt",
        role: "primary",
        modality: "optical",
        name: "Sentinel-2 MSI L2A",
        src: "/missions/kosi-optical.jpg",
        sensor: "S2-MSI",
        date: "18 Jul 2020",
        gsd: "10 m",
        location: "Kosi fan, Bihar",
        coords: "25.87° N, 86.60° E",
        format: "GeoTIFF",
        crs: "EPSG:4326",
      },
      {
        id: "kosi-sar",
        role: "secondary",
        modality: "sar",
        name: "Sentinel-1 IW GRD VV",
        src: "/missions/kosi-sar.jpg",
        sensor: "S1-IW",
        date: "18 Jul 2020",
        gsd: "10 m",
        location: "Kosi fan, Bihar",
        coords: "25.87° N, 86.60° E",
        format: "GeoTIFF",
        crs: "EPSG:4326",
      },
    ],
    suggested: [
      "Identify flooded areas in this region and show the affected settlements.",
      "Use the optical and SAR images together to identify built-up and water-covered regions.",
      "Highlight the water body referred to in the query.",
    ],
    replies: [
      { match: ["flood", "inundat", "settlement", "affected", "water-cover", "water covered", "built-up and water"], result: kosiFlood },
      { match: ["optical and sar", "sar", "together"], result: kosiFlood },
      { match: ["highlight", "water body", "water"], result: kosiFlood },
    ],
    fallback: kosiFlood,
  },
  {
    id: "mundra",
    code: "SQ-11",
    title: "Mundra Anchorage",
    location: "Kutch, Gujarat",
    synopsis: "A single VHR optical frame of a working terminal. Grounding and counting are the mandatory single-image proof.",
    demonstrates: "Single-image VQA · text-guided grounding · captioning",
    mode: "single",
    assets: [
      {
        id: "mundra",
        role: "primary",
        modality: "optical",
        name: "VHR Optical · Cartosat class",
        src: "/missions/mundra-port.jpg",
        sensor: "C2S-PAN/MX",
        date: "12 Nov 2023",
        gsd: "0.6 m",
        location: "Mundra Port, Gujarat",
        coords: "22.738° N, 69.708° E",
        format: "GeoTIFF",
        crs: "EPSG:4326",
      },
    ],
    suggested: [
      "Highlight the ships in this harbour.",
      "How many ships are visible, and where are the storage tanks?",
      "Describe the land-cover and major objects visible in this image.",
    ],
    replies: [
      { match: ["ship", "harbour", "harbor", "vessel", "tank"], result: mundraGround },
      { match: ["how many"], result: mundraVqa },
      { match: ["describe", "land-cover", "land cover", "caption"], result: mundraCaption },
    ],
    fallback: mundraCaption,
  },
  {
    id: "gurgaon",
    code: "SQ-19",
    title: "Gurugram Expansion",
    location: "Gurugram, Haryana",
    synopsis: "Thirty-three years apart. The only legal answer to a change question is increase, decrease, or unchanged — with a map.",
    demonstrates: "Bi-temporal change VQA · change mask · description",
    mode: "bi-temporal",
    assets: [
      {
        id: "gg-then",
        role: "primary",
        modality: "optical",
        name: "Landsat TM · 1991",
        src: "/missions/gurgaon-1991.jpg",
        sensor: "L5-TM",
        date: "05 Sep 1991",
        gsd: "30 m",
        location: "Gurugram",
        coords: "28.46° N, 77.03° E",
        format: "GeoTIFF",
        crs: "EPSG:4326",
      },
      {
        id: "gg-now",
        role: "secondary",
        modality: "optical",
        name: "VHR Optical · 2024",
        src: "/missions/gurgaon-now.jpg",
        sensor: "C2S-PAN/MX",
        date: "15 Apr 2024",
        gsd: "0.6 m",
        location: "Gurugram",
        coords: "28.46° N, 77.03° E",
        format: "GeoTIFF",
        crs: "EPSG:4326",
      },
    ],
    suggested: [
      "Has the built-up area increased, decreased, or remained unchanged?",
      "What changed between these two dates, and where did the change occur?",
      "Describe the urban expansion.",
    ],
    replies: [
      { match: ["built-up", "built up", "increased", "decreased", "unchanged", "change", "expansion", "between"], result: gurgaonChange },
    ],
    fallback: gurgaonChange,
  },
  {
    id: "doaba",
    code: "SQ-23",
    title: "Doaba Mosaic",
    location: "Ludhiana belt, Punjab",
    synopsis: "The quiet dossier: land-cover captioning and VQA on a single MSI chip. This is the baseline the problem statement calls mandatory.",
    demonstrates: "Single-image captioning · VQA · land-cover",
    mode: "single",
    assets: [
      {
        id: "doaba",
        role: "primary",
        modality: "optical",
        name: "Sentinel-2 MSI L2A",
        src: "/missions/punjab-fields.jpg",
        sensor: "S2-MSI",
        date: "10 Feb 2024",
        gsd: "10 m",
        location: "Doaba, Punjab",
        coords: "30.84° N, 75.88° E",
        format: "GeoTIFF",
        crs: "EPSG:4326",
      },
    ],
    suggested: [
      "Describe the land-cover and major objects visible in this image.",
      "Is agriculture the dominant class in this scene?",
      "Where is the urban core?",
    ],
    replies: [
      { match: ["describe", "land-cover", "land cover", "caption", "objects"], result: punjabCaption },
      { match: ["agriculture", "dominant", "crop"], result: punjabVqa },
      { match: ["urban", "where", "core", "highlight"], result: punjabCaption },
    ],
    fallback: punjabCaption,
  },
];

export function missionById(id: string) {
  return missions.find((m) => m.id === id);
}

export function customMission(files: { name: string; src: string }[]): Mission {
  const pair = files.length >= 2;
  const mode: Mission["mode"] = pair ? "cross-modal" : "single";
  const assets = files.slice(0, 2).map((f, i) => ({
    id: `up-${i}`,
    role: (i === 0 ? "primary" : "secondary") as "primary" | "secondary",
    modality: (i === 1 && pair ? "sar" : "optical") as "optical" | "sar",
    name: f.name,
    src: f.src,
    sensor: "User scene",
    date: "Uploaded",
    gsd: "native",
    location: "Local file",
    coords: "from metadata",
    format: f.name.split(".").pop()?.toUpperCase() || "FILE",
    crs: "from GeoTIFF tags",
  }));

  const fallback: QueryResult = {
    task: pair ? "cross-modal" : "vqa",
    title: "Scene interrogation",
    answer:
      "Inputs are accepted and routed. This prototype UI is running the agentic controller against a local specialist registry. Connect the fine-tuned RSVQA / grounding / CDVQA weights to replace this brief with model output — the canvas, evidence, and audit trail stay the same.",
    observations: [
      "Compatibility check completed on the uploaded file set.",
      "Task routing selected from query + input cardinality.",
      "Visual evidence layer is ready for model masks and boxes.",
    ],
    metrics: [
      { label: "Inputs", value: String(assets.length) },
      { label: "Mode", value: mode },
      { label: "Registry", value: "4 specialists" },
    ],
    confidence: 0.5,
    boxes: [],
    masks: [],
    layers: [],
    models: [
      { name: "Input Guardian", role: "upload inspection", status: "complete" },
      { name: "RSVQA-Adapter", role: "awaiting weights", status: "standby" },
    ],
    trace: [
      { id: "v", label: "Input validation", detail: `${assets.length} file(s) · format accepted`, status: "pending" },
      { id: "m", label: "Modality detection", detail: pair ? "Treated as pair — confirm optical/SAR or t₀/t₁" : "Single frame", status: "pending" },
      { id: "q", label: "Query parse", detail: "Intent classified from text", status: "pending" },
      { id: "t", label: "Tool selection", detail: "Registry routed", status: "pending" },
      { id: "e", label: "Visual evidence", detail: "Awaiting specialist tensors", status: "pending" },
      { id: "i", label: "Integration", detail: "Controller ready", status: "pending" },
    ],
  };

  return {
    id: "upload",
    code: "SQ-XX",
    title: "Local scene",
    location: "Uploaded",
    synopsis: "User-supplied frame(s).",
    demonstrates: "Upload · compatibility · routing",
    mode,
    assets,
    suggested: [
      "Describe the land-cover and major objects visible in this image.",
      "What is the dominant class in this scene?",
      pair ? "Use both images together to identify built-up and water-covered regions." : "Highlight the largest water body.",
    ],
    replies: [],
    fallback,
  };
}
