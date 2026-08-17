const HF_MODULE = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const CLIP_MODEL = 'Xenova/clip-vit-base-patch32';
const SAM_MODEL = 'Xenova/slimsam-77-uniform';

let hf = null;
let visionModel = null;
let visionProcessor = null;
let samModel = null;
let samProcessor = null;
let device = 'wasm';
let referenceEmbeddings = [];
let initPromise = null;
let samPromise = null;

function normalize(vec) {
  let n = 0;
  for (const v of vec) n += v * v;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(vec, v => v / n);
}

function cosine(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

function clamp(v) { return Math.max(0, Math.min(1, v)); }

async function loadHF() {
  if (!hf) hf = await import(HF_MODULE);
  return hf;
}

async function loadVisionWithFallback(lib, progress) {
  const attempts = navigator.gpu
    ? [
        { device: 'webgpu', dtype: 'q4f16', label: 'WebGPU q4f16' },
        { device: 'wasm', dtype: 'q8', label: 'WASM q8 fallback' },
      ]
    : [{ device: 'wasm', dtype: 'q8', label: 'WASM q8' }];
  let lastError;
  for (const a of attempts) {
    try {
      progress({ phase: 'runtime', text: `Trying ${a.label}` });
      const model = await lib.CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL, {
        device: a.device,
        dtype: a.dtype,
        progress_callback: p => progress({ phase: 'model', detail: p }),
      });
      device = a.device;
      return model;
    } catch (e) {
      lastError = e;
      progress({ phase: 'runtime-fallback', text: `${a.label} unavailable — trying fallback` });
      console.warn('BallHunter AI runtime attempt failed:', a, e);
    }
  }
  throw lastError || new Error('No supported neural runtime found');
}

async function init(progress = () => {}) {
  if (visionModel && visionProcessor) return { device };
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const lib = await loadHF();
    visionProcessor = await lib.AutoProcessor.from_pretrained(CLIP_MODEL, {
      progress_callback: p => progress({ phase: 'processor', detail: p }),
    });
    visionModel = await loadVisionWithFallback(lib, progress);
    progress({ phase: 'ready', text: `Neural re-ID ready · ${device.toUpperCase()}` });
    return { device };
  })();
  try { return await initPromise; }
  catch (e) { initPromise = null; visionModel = null; throw e; }
}

async function embedCanvas(canvas) {
  await init();
  const image = hf.RawImage.fromCanvas(canvas);
  const inputs = await visionProcessor(image);
  const out = await visionModel(inputs);
  const tensor = out.image_embeds || out.pooler_output || Object.values(out).find(v => v?.data && v?.dims);
  if (!tensor?.data) throw new Error('Vision model did not return an embedding');
  return normalize(tensor.data);
}

async function setReferenceCanvases(canvases, progress = () => {}) {
  await init(progress);
  referenceEmbeddings = [];
  for (let i = 0; i < canvases.length; i++) {
    progress({ phase: 'reference', text: `Learning reference view ${i + 1}/${canvases.length}` });
    referenceEmbeddings.push(await embedCanvas(canvases[i]));
  }
  progress({ phase: 'reference-ready', text: `${referenceEmbeddings.length} neural reference view${referenceEmbeddings.length === 1 ? '' : 's'} ready` });
  return referenceEmbeddings.length;
}

async function similarity(canvas) {
  if (!referenceEmbeddings.length) throw new Error('Reference embeddings are not ready');
  const e = await embedCanvas(canvas);
  const scores = referenceEmbeddings.map(r => cosine(e, r)).sort((a, b) => b - a);
  const raw = scores.length > 1 ? scores[0] * 0.72 + scores[1] * 0.28 : scores[0];
  return { raw, score: clamp((raw - 0.28) / 0.56) };
}

async function loadSamWithFallback(lib, progress) {
  const attempts = device === 'webgpu'
    ? [
        { device: 'webgpu', dtype: 'fp16', label: 'WebGPU fp16' },
        { device: 'wasm', dtype: 'q8', label: 'WASM q8 fallback' },
      ]
    : [{ device: 'wasm', dtype: 'q8', label: 'WASM q8' }];
  let lastError;
  for (const a of attempts) {
    try {
      progress({ phase: 'sam-runtime', text: `Mask refiner: ${a.label}` });
      const model = await lib.SamModel.from_pretrained(SAM_MODEL, {
        device: a.device,
        dtype: a.dtype,
        progress_callback: p => progress({ phase: 'sam-model', detail: p }),
      });
      return model;
    } catch (e) {
      lastError = e;
      console.warn('BallHunter SlimSAM runtime attempt failed:', a, e);
    }
  }
  throw lastError || new Error('No supported SlimSAM runtime found');
}

async function initSegmenter(progress = () => {}) {
  if (samModel && samProcessor) return true;
  if (samPromise) return samPromise;
  samPromise = (async () => {
    const lib = await loadHF();
    progress({ phase: 'sam-loading', text: 'Loading neural mask refiner' });
    samProcessor = await lib.AutoProcessor.from_pretrained(SAM_MODEL, {
      progress_callback: p => progress({ phase: 'sam-processor', detail: p }),
    });
    samModel = await loadSamWithFallback(lib, progress);
    progress({ phase: 'sam-ready', text: 'SlimSAM mask refiner ready' });
    return true;
  })();
  try { return await samPromise; }
  catch (e) { samPromise = null; samModel = null; throw e; }
}

async function segmentCandidate(canvas, progress = () => {}) {
  await initSegmenter(progress);
  const raw = hf.RawImage.fromCanvas(canvas);
  const input_points = [[[canvas.width / 2, canvas.height / 2]]];
  const inputs = await samProcessor(raw, { input_points });
  const outputs = await samModel(inputs);
  const masks = await samProcessor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
  const t = masks?.[0];
  if (!t?.data || !t?.dims) throw new Error('Mask model returned no masks');
  const dims = t.dims;
  const H = dims[dims.length - 2], W = dims[dims.length - 1];
  const stride = H * W;
  const maskCount = Math.max(1, Math.floor(t.data.length / stride));
  const iou = outputs.iou_scores?.data || [];
  let best = 0;
  for (let i = 1; i < maskCount; i++) if ((iou[i] ?? 0) > (iou[best] ?? 0)) best = i;
  const off = best * stride;
  let area = 0, minX = W, minY = H, maxX = -1, maxY = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (t.data[off + y * W + x] > 0) {
      area++; minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
  }
  if (!area || maxX < 0) return { score: 0, areaRatio: 0, aspect: 0, fill: 0, iou: iou[best] ?? 0 };
  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const aspect = Math.min(bw, bh) / Math.max(bw, bh);
  const fill = area / (bw * bh);
  const areaRatio = area / stride;
  const score = clamp(0.52 * aspect + 0.28 * clamp(fill / 0.72) + 0.20 * clamp((iou[best] ?? 0.5)));
  return { score, areaRatio, aspect, fill, iou: iou[best] ?? 0 };
}

export const BallHunterAI = {
  init,
  setReferenceCanvases,
  similarity,
  segmentCandidate,
  get status() { return { ready: !!visionModel, samReady: !!samModel, device, references: referenceEmbeddings.length }; },
};
