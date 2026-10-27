<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Project rules

- Face-swap runs fully in the browser: uploads stay as object URLs, all four models are preloaded on page open via `preloadModels` (Start stays disabled until ready), and "Start" calls `swapFaces` from `src/lib/faceSwap.js`; why: the user asked for no backend or database.
- Face-swap UI pieces live in `src/components/face-swap/`; why: keeps the single-page tool's parts together.
- The app is dark-only (tokens defined in `:root`, no `.dark` toggle); why: the requested design is a dark theme.
- Face-swap AI logic lives in `src/lib/faceSwap.js`, imports `onnxruntime-web/webgpu` (not the default entry), and is loaded only via dynamic `import()` on the client; why: the user requested WebGPU, and ORT uses browser APIs that break SSR.
- `onnxruntime-web` is excluded from Vite `optimizeDeps`; why: pre-bundling breaks its `new URL(..., import.meta.url)` wasm lookup in dev.
- SCRFD detector decoding lives in `src/lib/scrfd.js` as pure, DOM-free functions that classify outputs by shape, not name; why: lets it be tested in Node against the real model and survives exports with different output names.
- Face alignment math (similarity transform to the ArcFace template) lives in `src/lib/faceAlign.js`, DOM-free with tests in `faceAlign.test.ts`; why: keeps the warp/inverse-warp verifiable outside the browser.
- Face-swap output is encoded with WebCodecs `VideoEncoder` (H.264, falling back to VP9) and muxed into MP4 with `mp4-muxer`, one frame at a time inside the loop; why: faster than MediaRecorder and no per-frame storage.
- DOM-free image/video helpers (working-resolution fit, face-box motion check, WASM thread count) live in `src/lib/imageOps.js` with tests in `imageOps.test.ts`; why: keeps them verifiable outside the browser.
- The detector runs on every frame and its keypoints always place the pasted face. The swapper (and GFPGAN) are skipped only when `canReuseSwap` in `imageOps.js` allows it (small motion vs the previous frame, capped consecutive reuses); then the cached swap pixels are pasted at the current frame's keypoints. No cross-frame blending, and a frame with no face is left untouched and clears the cache; why: removes flicker while keeping the face locked to the real position and the mouth moving.
- Pages are served cross-origin isolated (COOP `same-origin` + COEP `credentialless`) via a dev-server middleware plugin in `vite.config.ts`, `src/server.ts`, and `public/_headers`; why: multi-threaded WASM needs SharedArrayBuffer, and the sandbox strips Vite `server.headers`.
- Optional GFPGAN restoration (`highQuality` option of `swapFaces`) re-crops the swapped face from the frame at 512x512 with the FFHQ template (`ffhqMatrix` in `faceAlign.js`), not by stretching the 128 swapper output; why: GFPGAN was trained on FFHQ-aligned faces. It is preloaded with the other models, tries WebGPU and falls back to WASM.
