// Face-swap AI logic (runs entirely in the browser).
//
// Uses the WebGPU build of ONNX Runtime Web. This module touches browser-only
// APIs, so load it lazily from an event handler or useEffect — never import it
// at the top of a route file:
//
//   const { loadModel, runModel } = await import("@/lib/faceSwap.js");
//
import * as ort from "onnxruntime-web/webgpu";

/** True when the current browser exposes WebGPU. */
export function isWebGpuAvailable() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

/**
 * Create an inference session for an .onnx model.
 * @param {string | ArrayBuffer | Uint8Array} model URL or raw bytes of the model
 * @returns {Promise<ort.InferenceSession>}
 */
export async function loadModel(model) {
  const executionProviders = isWebGpuAvailable() ? ["webgpu", "wasm"] : ["wasm"];
  return ort.InferenceSession.create(model, { executionProviders });
}

/**
 * Run a session with a map of input tensors.
 * @param {ort.InferenceSession} session
 * @param {Record<string, ort.Tensor>} feeds
 */
export async function runModel(session, feeds) {
  return session.run(feeds);
}

/**
 * Swap the face from `faceImage` onto every frame of `video`.
 * TODO: add face detection, embedding, swapping and frame re-encoding here.
 * @param {File} faceImage
 * @param {File} video
 * @param {(progress: number) => void} [onProgress]
 * @returns {Promise<Blob>} the processed video
 */
export async function swapFaces(faceImage, video, onProgress) {
  void faceImage;
  onProgress?.(100);
  return video;
}

export { ort };
