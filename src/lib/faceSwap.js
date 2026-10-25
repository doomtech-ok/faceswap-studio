// Face-swap AI logic (runs entirely in the browser).
//
// Uses the WebGPU build of ONNX Runtime Web. This module touches browser-only
// APIs, so load it lazily from an event handler or useEffect — never import it
// at the top of a route file:
//
//   const { swapFaces } = await import("@/lib/faceSwap.js");
//
import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.simd = true;
ort.env.wasm.numThreads = 1;

import { extractEmap, projectEmbedding } from './onnxEmap.js';
import { DET_SIZE, rgbaToDetInput, decodeScrfd } from './scrfd.js';
import { alignMatrix, invertMatrix } from './faceAlign.js';

const SWAPPER_URL =
  'https://huggingface.co/datasets/JCTN/ReActor/resolve/main/models/inswapper_128.onnx';
// SCRFD 10G con keypoints (det_10g de InsightFace buffalo_l), ~17 MB.
const DETECTOR_URL = 'https://huggingface.co/maze/faceX/resolve/main/det_10g.onnx';
const RECOGNIZER_URL = 'https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx';

let detectorSession = null;
let swapperSession = null;
let recognizerSession = null;
let emap = null; // Float32Array 512x512 (fila mayor)

export { ort };

export function isWebGpuAvailable() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

// Pass `executionProviders` to force a backend (e.g. ['wasm']); by default
// WebGPU is used when available, falling back to WASM.
export async function loadModel(modelUrl, executionProviders) {
  return await ort.InferenceSession.create(modelUrl, {
    executionProviders:
      executionProviders ?? (isWebGpuAvailable() ? ['webgpu', 'wasm'] : ['wasm']),
  });
}

export async function runModel(session, feeds) {
  return await session.run(feeds);
}

// Carga los tres modelos de forma secuencial (WebGPU solo permite crear una
// sesión a la vez) y bloquea cargas concurrentes.
let isLoading = false;
let loadingPromise = null;

async function ensureModels() {
  if (detectorSession && swapperSession && recognizerSession) {
    return { detectorSession, swapperSession, recognizerSession };
  }
  // Si ya hay una carga en curso, esperamos esa en lugar de crear otra sesión.
  if (isLoading && loadingPromise) return loadingPromise;

  isLoading = true;
  loadingPromise = (async () => {
    try {
      if (!detectorSession) {
        console.log('Cargando detector facial (SCRFD)...');
        detectorSession = await loadModel(DETECTOR_URL);
        console.log('✅ Detector listo');
      }
      if (!swapperSession) {
        console.log('Descargando swapper (puede tardar por el tamaño)...');
        // Descargamos los bytes nosotros mismos para leer la matriz emap del
        // archivo (ORT Web no expone los initializers de la sesión).
        const res = await fetch(SWAPPER_URL);
        if (!res.ok) throw new Error(`No se pudo descargar el swapper (${res.status}).`);
        let bytes = new Uint8Array(await res.arrayBuffer());
        emap = extractEmap(bytes);
        console.log('✅ Matriz emap extraída [512,512]');
        // Precisión completa y solo WASM: la RX 580 no soporta shader-f16 en WebGPU.
        swapperSession = await loadModel(bytes, ['wasm']);
        bytes = null; // liberar memoria
        console.log('✅ Swapper listo');
      }
      if (!recognizerSession) {
        console.log('Cargando reconocedor facial...');
        recognizerSession = await loadModel(RECOGNIZER_URL);
        console.log('✅ Reconocedor listo');
      }
      return { detectorSession, swapperSession, recognizerSession };
    } finally {
      isLoading = false;
      loadingPromise = null;
    }
  })();
  return loadingPromise;
}

// --- Detección de rostros (SCRFD) ---

let detCanvas = null;

// Devuelve el rostro con mayor puntuación en `source` (canvas, video o bitmap)
// como { x, y, width, height, score, kps } en píxeles de `source`, o null.
async function detectFace(source, srcW, srcH) {
  if (!detCanvas) {
    detCanvas = document.createElement('canvas');
    detCanvas.width = DET_SIZE;
    detCanvas.height = DET_SIZE;
  }
  const ctx = detCanvas.getContext('2d', { willReadFrequently: true });
  // Letterbox: escalar manteniendo proporción y pegar arriba a la izquierda.
  const scale = Math.min(DET_SIZE / srcW, DET_SIZE / srcH);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, DET_SIZE, DET_SIZE);
  ctx.drawImage(source, 0, 0, srcW * scale, srcH * scale);

  const rgba = ctx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
  const tensor = new ort.Tensor('float32', rgbaToDetInput(rgba), [1, 3, DET_SIZE, DET_SIZE]);
  const out = await detectorSession.run({ [detectorSession.inputNames[0]]: tensor });
  const outputs = detectorSession.outputNames.map((n) => out[n]);
  const faces = decodeScrfd(outputs, scale);
  return faces[0] ?? null;
}

// Recorta y alinea el rostro: dibuja `source` en `outCtx` (size x size)
// aplicando la transformación de similitud que lleva los 5 keypoints a la
// plantilla ArcFace. Ojos y boca quedan horizontales y sin deformación.
function warpAligned(outCtx, source, matrix, size) {
  outCtx.setTransform(1, 0, 0, 1, 0, 0);
  outCtx.clearRect(0, 0, size, size);
  outCtx.setTransform(...matrix);
  outCtx.imageSmoothingQuality = 'high';
  outCtx.drawImage(source, 0, 0);
  outCtx.setTransform(1, 0, 0, 1, 0, 0);
}

// Pega `face` (alineado, 128x128) de vuelta en el frame con la transformación
// inversa, usando una máscara con bordes suavizados para que no se note el corte.
let pasteCanvas = null;
function pasteAligned(ctx, face, matrix) {
  const size = face.width;
  if (!pasteCanvas) {
    pasteCanvas = document.createElement('canvas');
    pasteCanvas.width = size;
    pasteCanvas.height = size;
  }
  const t = pasteCanvas.getContext('2d');
  t.globalCompositeOperation = 'source-over';
  t.clearRect(0, 0, size, size);
  t.drawImage(face, 0, 0);
  const r = size / 2;
  const grad = t.createRadialGradient(r, r, r * 0.65, r, r, r);
  grad.addColorStop(0, 'rgba(0,0,0,1)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  t.globalCompositeOperation = 'destination-in';
  t.fillStyle = grad;
  t.fillRect(0, 0, size, size);

  ctx.save();
  ctx.setTransform(...invertMatrix(matrix));
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(pasteCanvas, 0, 0);
  ctx.restore();
}

// Extrae el embedding facial [1,512] con el reconocedor (ArcFace w600k_r50)
// a partir del rostro alineado a 112x112, y lo proyecta con la matriz emap.
async function getFaceEmbedding(source, kps) {
  const canvas = document.createElement('canvas');
  canvas.width = 112;
  canvas.height = 112;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  warpAligned(ctx, source, alignMatrix(kps, 112), 112);

  const imgData = ctx.getImageData(0, 0, 112, 112).data;
  const plane = 112 * 112;
  const floatData = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i++) {
    // Normalización de insightface para w600k_r50: (x - 127.5) / 127.5, RGB.
    floatData[i] = (imgData[i * 4] - 127.5) / 127.5; // R
    floatData[i + plane] = (imgData[i * 4 + 1] - 127.5) / 127.5; // G
    floatData[i + 2 * plane] = (imgData[i * 4 + 2] - 127.5) / 127.5; // B
  }

  const tensor = new ort.Tensor('float32', floatData, [1, 3, 112, 112]);
  // Usamos el nombre real de la entrada del modelo (suele ser 'input.1' o 'input').
  const inputName = recognizerSession.inputNames[0] ?? 'input';
  const output = await recognizerSession.run({ [inputName]: tensor });
  const raw = output[recognizerSession.outputNames[0] ?? Object.keys(output)[0]];

  // result[j] = sum_i(embedding[i] * emap[i][j]), con normalización L2 como insightface.
  const projected = projectEmbedding(raw.data, emap);
  return new ort.Tensor('float32', projected, [1, 512]);
}

// --- Utilidades de imagen ---

// Returns a planar NCHW tensor [1, 3, size, size] with RGB values in [0, 1],
// which is the layout and range inswapper_128 expects for its "target" input.
function canvasToTensor(canvas, size) {
  const ctx = canvas.getContext('2d');
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imgData.data;
  const plane = size * size;
  const floatData = new Float32Array(plane * 3);
  const scale = canvas.width / size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const srcX = Math.floor(x * scale);
      const srcY = Math.floor(y * scale);
      const idx = (srcY * canvas.width + srcX) * 4;
      const p = y * size + x;
      floatData[p] = data[idx] / 255;
      floatData[plane + p] = data[idx + 1] / 255;
      floatData[2 * plane + p] = data[idx + 2] / 255;
    }
  }
  return new ort.Tensor('float32', floatData, [1, 3, size, size]);
}

// Reads a planar NCHW [1, 3, size, size] output in [0, 1] into `outCanvas`
// (reused across frames so no new canvas is created per frame).
function tensorToCanvas(outData, size, outCanvas) {
  const outCtx = outCanvas.getContext('2d');
  const imgData = outCtx.createImageData(size, size);
  const plane = size * size;
  for (let j = 0; j < plane; j++) {
    imgData.data[j * 4] = outData[j] * 255;
    imgData.data[j * 4 + 1] = outData[plane + j] * 255;
    imgData.data[j * 4 + 2] = outData[2 * plane + j] * 255;
    imgData.data[j * 4 + 3] = 255;
  }
  outCtx.putImageData(imgData, 0, 0);
  return outCanvas;
}

// Guarda el frame como JPEG comprimido (~100 KB) en lugar de ImageData sin
// comprimir (~8 MB en 1080p), que era lo que agotaba la memoria.
function canvasToBlob(canvas) {
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('No se pudo guardar el fotograma.'))),
      'image/jpeg',
      0.92
    )
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function pickRecorderMimeType() {
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) ?? '';
}

// --- Procesamiento de video ---

/**
 * @param {File | Blob} faceImage source face image
 * @param {File | Blob} videoFile target video
 * @param {(fraction: number) => void} [onProgress] called with 0..1
 * @returns {Promise<Blob>} the processed video
 */
export async function swapFaces(faceImage, videoFile, onProgress) {
  // El detector se carga primero; los tres modelos quedan listos antes del bucle.
  const { swapperSession } = await ensureModels();

  // 1. Preparar el rostro fuente: detectar la cara en la foto, alinearla con
  // sus 5 keypoints y sacar el embedding [1,512] solo de ese rostro.
  // drawImage no acepta un File, así que lo convertimos a bitmap primero.
  const sourceBitmap = await createImageBitmap(faceImage);
  let sourceEmbedding;
  try {
    const sourceFace = await detectFace(sourceBitmap, sourceBitmap.width, sourceBitmap.height);
    if (!sourceFace || sourceFace.kps.length !== 5) {
      throw new Error('No se encontró un rostro en la foto de origen. Prueba con una foto más clara y de frente.');
    }
    console.log('Rostro detectado en la foto de origen:', sourceFace);
    sourceEmbedding = await getFaceEmbedding(sourceBitmap, sourceFace.kps);
  } finally {
    sourceBitmap.close();
  }

  // 2. Decodificar el video: <video> + canvas (más compatible que WebCodecs,
  // que además necesitaría un demuxer para leer el archivo).
  const videoEl = document.createElement('video');
  videoEl.muted = true;
  videoEl.preload = 'auto';
  videoEl.src = URL.createObjectURL(videoFile);

  try {
    await new Promise((resolve, reject) => {
      videoEl.onloadedmetadata = resolve;
      videoEl.onerror = () => reject(new Error('No se pudo leer el video.'));
    });

    const duration = videoEl.duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      throw new Error('El video no tiene una duración válida.');
    }

    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const fps = 10; // procesamos 10 fps para no saturar la RX 580

    // Canvas temporales: se crean una sola vez y se reutilizan en cada frame.
    const faceCanvas = document.createElement('canvas');
    faceCanvas.width = 128;
    faceCanvas.height = 128;
    const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true });
    const swapOutCanvas = document.createElement('canvas');
    swapOutCanvas.width = 128;
    swapOutCanvas.height = 128;

    // Mueve el console.log de las entradas AQUÍ, antes del bucle
    console.log('Entradas que espera el modelo:', swapperSession.inputNames);

    const CHUNK_SIZE = 10; // fotogramas por bloque
    const frameBlobs = [];
    // Procesar el video completo a `fps` cuadros por segundo, en bloques.
    const totalFrames = Math.max(1, Math.floor(duration * fps));
    const totalChunks = Math.ceil(totalFrames / CHUNK_SIZE);

    for (let chunk = 0; chunk < totalChunks; chunk++) {
      const start = chunk * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, totalFrames);

      for (let i = start; i < end; i++) {
        const seeked = new Promise((r) => (videoEl.onseeked = r));
        videoEl.currentTime = i / fps;
        await seeked;

        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

        // Detectar el rostro con SCRFD. Si no hay rostro, el frame queda igual.
        const face = await detectFace(canvas, canvas.width, canvas.height);
        if (face && face.kps.length === 5) {
          if (i === 0) console.log('Rostro detectado en el frame 0:', face);
          // Alinear el rostro a 128x128 con los 5 keypoints (plantilla ArcFace).
          const matrix = alignMatrix(face.kps, 128);
          warpAligned(faceCtx, canvas, matrix, 128);

          // Ejecutar el swapper: inswapper_128 usa "target" [1,3,128,128] y "source".
          const targetTensor = canvasToTensor(faceCanvas, 128);

          let output;
          try {
            output = await swapperSession.run({
              target: targetTensor,
              source: sourceEmbedding, // [1, 512]
            });
          } catch (e) {
            throw new Error(
              'El swapper rechazó las entradas. inswapper_128 espera "source" como un ' +
                'embedding facial [1,512] (de un modelo de reconocimiento como ArcFace), ' +
                `no una imagen. Detalle: ${e instanceof Error ? e.message : String(e)}`
            );
          }

          // Tomar la salida y devolverla al frame con la transformación inversa.
          const outKey = swapperSession.outputNames[0] ?? Object.keys(output)[0];
          tensorToCanvas(output[outKey].data, 128, swapOutCanvas);
          pasteAligned(ctx, swapOutCanvas, matrix);
          // Liberar los tensores de este frame.
          targetTensor.dispose?.();
          for (const t of Object.values(output)) t.dispose?.();
        } else {
          console.warn(`Frame ${i}: no se detectó ningún rostro, se deja sin cambios.`);
        }

        frameBlobs.push(await canvasToBlob(canvas));

        // Al final de cada iteración, cede el hilo
        await sleep(100);

        if (onProgress) onProgress((i + 1) / totalFrames);
      }

      // Fin del bloque: limpiar los canvas temporales y dar un respiro mayor.
      faceCtx.clearRect(0, 0, 128, 128);
      swapOutCanvas.getContext('2d').clearRect(0, 0, 128, 128);
      console.log(`Bloque ${chunk + 1} procesado (${end}/${totalFrames} fotogramas)`);
      await sleep(300);
    }

    // 3. Codificar los frames a un nuevo video
    const mimeType = pickRecorderMimeType();
    const stream = canvas.captureStream(fps);
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    const done = new Promise((r) => (recorder.onstop = r));

    recorder.start();
    // Decodificamos el siguiente JPEG mientras se muestra el actual, y soltamos
    // cada uno en cuanto se dibuja.
    let next = createImageBitmap(frameBlobs[0]);
    for (let i = 0; i < frameBlobs.length; i++) {
      const bitmap = await next;
      frameBlobs[i] = null;
      next = i + 1 < frameBlobs.length ? createImageBitmap(frameBlobs[i + 1]) : null;
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
      await sleep(1000 / fps);
    }
    recorder.stop();
    await done;

    return new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
  } finally {
    URL.revokeObjectURL(videoEl.src);
  }
}
