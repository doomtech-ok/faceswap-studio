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

- Face-swap runs fully in the browser: uploads stay as object URLs and "Start" calls `swapFaces` from `src/lib/faceSwap.js`; why: the user asked for no backend or database.
- Face-swap UI pieces live in `src/components/face-swap/`; why: keeps the single-page tool's parts together.
- The app is dark-only (tokens defined in `:root`, no `.dark` toggle); why: the requested design is a dark theme.
- Face-swap AI logic lives in `src/lib/faceSwap.js`, imports `onnxruntime-web/webgpu` (not the default entry), and is loaded only via dynamic `import()` on the client; why: the user requested WebGPU, and ORT uses browser APIs that break SSR.
- `onnxruntime-web` is excluded from Vite `optimizeDeps`; why: pre-bundling breaks its `new URL(..., import.meta.url)` wasm lookup in dev.
- SCRFD detector decoding lives in `src/lib/scrfd.js` as pure, DOM-free functions that classify outputs by shape, not name; why: lets it be tested in Node against the real model and survives exports with different output names.
- Face alignment math (similarity transform to the ArcFace template) lives in `src/lib/faceAlign.js`, DOM-free with tests in `faceAlign.test.ts`; why: keeps the warp/inverse-warp verifiable outside the browser.
