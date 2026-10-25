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

let detectorSession = null;
let swapperSession = null;

export { ort };

export function isWebGpuAvailable() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

export async function loadModel(modelUrl) {
  return await ort.InferenceSession.create(modelUrl, {
    executionProviders: isWebGpuAvailable() ? ['webgpu', 'wasm'] : ['wasm'],
  });
}

export async function runModel(session, feeds) {
  return await session.run(feeds);
}

// Inicializa solo el swapper. WebGPU solo permite crear una sesión a la vez,
// así que el detector queda fuera por ahora y bloqueamos cargas concurrentes.
let isLoading = false;
let loadingPromise = null;

async function ensureModels() {
  if (swapperSession) return { swapperSession };
  // Si ya hay una carga en curso, esperamos esa en lugar de crear otra sesión.
  if (isLoading && loadingPromise) return loadingPromise;

  isLoading = true;
  loadingPromise = (async () => {
    try {
      console.log('Cargando swapper (puede tardar por el tamaño)...');
      swapperSession = await loadModel(
        'https://huggingface.co/datasets/JCTN/ReActor/resolve/main/models/inswapper_128_fp16.onnx'
      );
      console.log('✅ Swapper listo');
      return { swapperSession };
    } finally {
      isLoading = false;
      loadingPromise = null;
    }
  })();
  return loadingPromise;
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

// Reads a planar NCHW [1, 3, size, size] output in [0, 1] into a canvas.
function tensorToCanvas(outData, size) {
  const outCanvas = document.createElement('canvas');
  outCanvas.width = size;
  outCanvas.height = size;
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
  const { swapperSession } = await ensureModels();

  // 1. Preparar el rostro fuente (imagen subida por el usuario)
  const sourceBitmap = await createImageBitmap(faceImage);
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = 128;
  sourceCanvas.height = 128;
  const sourceCtx = sourceCanvas.getContext('2d');
  sourceCtx.drawImage(sourceBitmap, 0, 0, 128, 128);
  sourceBitmap.close();

  // Nota: el modelo inswapper espera un rostro alineado.
  // Por ahora usamos la imagen tal cual; si el resultado no es bueno,
  // más adelante añadimos alineación con el detector.

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

    const frames = [];
    const fps = 10; // procesamos 10 fps para no saturar la RX 580
    const totalFrames = Math.max(1, Math.floor(duration * fps));

    const faceCanvas = document.createElement('canvas');
    faceCanvas.width = 128;
    faceCanvas.height = 128;
    const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true });

    const sourceTensor = canvasToTensor(sourceCanvas, 128);

    for (let i = 0; i < totalFrames; i++) {
      const seeked = new Promise((r) => (videoEl.onseeked = r));
      videoEl.currentTime = i / fps;
      await seeked;

      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

      // Detectar rostro (simplificado: usamos el frame completo reducido)
      // En una versión completa aquí va la detección real con detectorSession.
      // Por ahora, recortamos el centro del frame como "rostro" para probar el pipeline.
      const cropX = canvas.width * 0.3;
      const cropY = canvas.height * 0.2;
      const cropW = canvas.width * 0.4;
      const cropH = canvas.height * 0.4;
      faceCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, 128, 128);

      // Ejecutar el swapper: inswapper_128 usa "target" [1,3,128,128] y "source".
      const targetTensor = canvasToTensor(faceCanvas, 128);

      let output;
      console.log('Entradas que espera el modelo:', swapperSession.inputNames);
      try {
        output = await swapperSession.run({
          target: targetTensor,
          source: sourceTensor,
        });
      } catch (e) {
        throw new Error(
          'El swapper rechazó las entradas. inswapper_128 espera "source" como un ' +
            'embedding facial [1,512] (de un modelo de reconocimiento como ArcFace), ' +
            `no una imagen. Detalle: ${e instanceof Error ? e.message : String(e)}`
        );
      }

      // Tomar la salida y dibujarla de vuelta
      const outKey = swapperSession.outputNames[0] ?? Object.keys(output)[0];
      const outCanvas = tensorToCanvas(output[outKey].data, 128);

      // Pegar el rostro procesado de vuelta en el frame
      ctx.drawImage(outCanvas, cropX, cropY, cropW, cropH);

      frames.push(ctx.getImageData(0, 0, canvas.width, canvas.height));

      if (onProgress) onProgress((i + 1) / totalFrames);
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
    for (const frameData of frames) {
      ctx.putImageData(frameData, 0, 0);
      await new Promise((r) => setTimeout(r, 1000 / fps));
    }
    recorder.stop();
    await done;

    return new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
  } finally {
    URL.revokeObjectURL(videoEl.src);
  }
}
