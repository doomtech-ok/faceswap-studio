import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, Film, Loader2, ScanFace, Sparkles, Download, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { UploadZone } from "@/components/face-swap/UploadZone";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Swapframe — Local Video Face Swap" },
      { name: "description", content: "Upload a face and a short clip to swap faces locally in your browser." },
      { property: "og:title", content: "Swapframe — Local Video Face Swap" },
      { property: "og:description", content: "Upload a face and a short clip to swap faces locally in your browser." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const MAX_IMAGE_MB = 10;
const MAX_VIDEO_MB = 100;

function useObjectUrl(file: Blob | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url;
}

type Status = "idle" | "processing" | "done" | "error";
type ModelStatus = "loading" | "ready" | "error";

const RESULT_FILENAME = "faceswap_result.mp4";

function useModelPreload() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>("loading");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelLabel, setModelLabel] = useState("Starting…");
  const [modelError, setModelError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    setModelStatus("loading");
    setModelError(null);
    (async () => {
      try {
        // Loaded on demand: the AI runtime only works in the browser.
        const { preloadModels } = await import("@/lib/faceSwap.js");
        await preloadModels((s: { fraction: number; label: string }) => {
          if (!active) return;
          setModelProgress(Math.round(s.fraction * 100));
          setModelLabel(s.label);
        });
        if (!active) return;
        setModelProgress(100);
        setModelStatus("ready");
      } catch (e) {
        if (!active) return;
        console.error("Model loading failed:", e);
        setModelError(e instanceof Error ? e.message : String(e));
        setModelStatus("error");
      }
    })();
    return () => {
      active = false;
    };
  }, [attempt]);

  return { modelStatus, modelProgress, modelLabel, modelError, retry: () => setAttempt((a) => a + 1) };
}

function Index() {
  const [face, setFace] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [highQuality, setHighQuality] = useState(false);
  const { modelStatus, modelProgress, modelLabel, modelError, retry } = useModelPreload();
  const modelsReady = modelStatus === "ready";
  // Each run gets an id so a stale run can't overwrite state after a reset.
  const runId = useRef(0);

  const faceUrl = useObjectUrl(face);
  const videoUrl = useObjectUrl(video);
  const resultUrl = useObjectUrl(result);

  const reset = () => {
    runId.current += 1;
    setStatus("idle");
    setProgress(0);
    setResult(null);
    setError(null);
  };

  const start = async () => {
    if (!face || !video || !modelsReady) return;
    const id = ++runId.current;
    setStatus("processing");
    setProgress(0);
    setResult(null);
    setError(null);
    try {
      const { swapFaces } = await import("@/lib/faceSwap.js");
      const blob = await swapFaces(
        face,
        video,
        (fraction) => {
          if (runId.current === id) setProgress(Math.round(fraction * 100));
        },
        { highQuality },
      );
      if (runId.current !== id) return;
      setResult(blob.type === "video/mp4" ? blob : new Blob([blob], { type: "video/mp4" }));
      setStatus("done");
    } catch (e) {
      if (runId.current !== id) return;
      console.error("Face swap failed:", e);
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  };

  const ready = Boolean(face && video) && modelsReady && status !== "processing";

  return (
    <main className="min-h-screen bg-grid">
      <div className="mx-auto flex max-w-6xl flex-col gap-10 px-5 py-10 md:py-16">
        <header className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-primary">
              <span className="size-1.5 rounded-full bg-primary" />
              Swapframe · runs locally
            </div>
            <div
              role="status"
              className="flex items-center gap-2 rounded-full border bg-surface px-3 py-1 font-mono text-[11px] text-muted-foreground"
            >
              {modelStatus === "loading" && (
                <>
                  <Loader2 className="size-3 animate-spin text-primary" />
                  Loading AI Models… {modelProgress}%
                </>
              )}
              {modelStatus === "ready" && (
                <>
                  <CheckCircle2 className="size-3 text-primary" />
                  AI models ready
                </>
              )}
              {modelStatus === "error" && (
                <>
                  <AlertTriangle className="size-3 text-destructive" />
                  AI models failed to load
                </>
              )}
            </div>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Video face swap</h1>
          <p className="max-w-xl text-sm text-muted-foreground">
            Pick a face, pick a short clip, and start. Your files stay on this device.
          </p>
        </header>

        <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div className="flex flex-col gap-6 rounded-2xl border bg-card/80 p-5 backdrop-blur md:p-6">
            <UploadZone
              step="01"
              label="Source Face"
              hint="Image"
              accept="image/png,image/jpeg,image/webp"
              icon={<ScanFace className="size-5" />}
              kind="image"
              file={face}
              previewUrl={faceUrl}
              onFile={(f) => {
                setFace(f);
                reset();
              }}
              onClear={() => {
                setFace(null);
                reset();
              }}
              validate={(f) => {
                if (!f.type.startsWith("image/")) return "Please choose an image file.";
                if (f.size > MAX_IMAGE_MB * 1024 * 1024) return `Image must be under ${MAX_IMAGE_MB} MB.`;
                return null;
              }}
            />
            <UploadZone
              step="02"
              label="Target Video"
              hint="Short clip"
              accept="video/mp4,video/webm"
              icon={<Film className="size-5" />}
              kind="video"
              file={video}
              previewUrl={videoUrl}
              onFile={(f) => {
                setVideo(f);
                reset();
              }}
              onClear={() => {
                setVideo(null);
                reset();
              }}
              validate={(f) => {
                if (!["video/mp4", "video/webm"].includes(f.type)) return "Please choose an MP4 or WebM video.";
                if (f.size > MAX_VIDEO_MB * 1024 * 1024) return `Video must be under ${MAX_VIDEO_MB} MB.`;
                return null;
              }}
            />

            <div className="flex items-center justify-between gap-4 rounded-lg border bg-surface px-4 py-3">
              <div className="flex flex-col gap-1">
                <Label htmlFor="hq-toggle" className="cursor-pointer">
                  Enable High Quality (GFPGAN)
                </Label>
                <span className="text-xs text-muted-foreground">
                  {highQuality
                    ? "Sharper face, much slower."
                    : "Fast mode: the face may look a little soft."}
                </span>
              </div>
              <Switch
                id="hq-toggle"
                checked={highQuality}
                onCheckedChange={setHighQuality}
                disabled={status === "processing"}
              />
            </div>

            <Button variant="action" size="xl" disabled={!ready} onClick={start} className="w-full">
              {modelStatus === "loading" ? <Loader2 className="animate-spin" /> : <Sparkles />}
              {status === "processing"
                ? "Swapping…"
                : modelStatus === "loading"
                  ? `Loading AI Models… ${modelProgress}%`
                  : modelStatus === "error"
                    ? "AI models unavailable"
                    : "Start Face Swap"}
            </Button>
            <p className="-mt-3 text-center text-xs text-muted-foreground">
              Only use faces and footage you have permission to use.
            </p>
            <div className="flex items-start gap-2 rounded-lg border border-primary/30 bg-primary/10 p-3 text-xs text-muted-foreground">
              <AlertTriangle className="mt-0.5 size-4 shrink-0 text-primary" />
              <span>
                Processing runs on your computer's processor, so it's slow — roughly a few seconds per frame (10 frames per
                second of video). Long videos can take a very long time; keep this tab open until it finishes.
              </span>
            </div>
          </div>

          <section className="flex flex-col gap-3">
            <header className="flex items-baseline justify-between">
              <h2 className="text-sm font-semibold tracking-tight">
                <span className="mr-2 font-mono text-xs text-primary">03</span>
                Result
              </h2>
              <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                {status === "idle" && (modelStatus === "loading" ? `Models ${modelProgress}%` : "Waiting")}
                {status === "processing" && `${progress}%`}
                {status === "done" && "Ready"}
                {status === "error" && "Failed"}
              </span>
            </header>

            <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-2xl border bg-surface">
              {status === "done" && resultUrl ? (
                <video key={resultUrl} src={resultUrl} controls autoPlay playsInline className="h-full w-full bg-background object-contain" />
              ) : status === "processing" ? (
                <div className="flex w-full max-w-xs flex-col items-center gap-4 px-6 text-center">
                  <ScanFace className="size-8 animate-pulse text-primary" />
                  <Progress value={progress} className="h-1.5 bg-muted [&>div]:bg-primary" />
                  <p className="font-mono text-xs text-muted-foreground">
                    {progress === 0 ? "Preparing your face photo…" : `Processing frames… ${progress}%`}
                  </p>
                </div>
              ) : status === "error" ? (
                <div className="flex max-w-md flex-col items-center gap-3 px-6 text-center">
                  <AlertTriangle className="size-8 text-destructive" />
                  <p className="text-sm font-medium">The face swap didn't finish.</p>
                  <p className="font-mono text-xs break-words text-muted-foreground">{error}</p>
                </div>
              ) : modelStatus === "loading" ? (
                <div className="flex w-full max-w-xs flex-col items-center gap-4 px-6 text-center">
                  <Loader2 className="size-8 animate-spin text-primary" />
                  <Progress value={modelProgress} className="h-1.5 bg-muted [&>div]:bg-primary" />
                  <div className="flex flex-col gap-1">
                    <p className="text-sm font-medium">Loading AI Models… {modelProgress}%</p>
                    <p className="font-mono text-xs text-muted-foreground">{modelLabel}</p>
                    <p className="text-xs text-muted-foreground">About 1.1 GB the first time — keep this tab open.</p>
                  </div>
                </div>
              ) : modelStatus === "error" ? (
                <div className="flex max-w-md flex-col items-center gap-3 px-6 text-center">
                  <AlertTriangle className="size-8 text-destructive" />
                  <p className="text-sm font-medium">The AI models couldn't load.</p>
                  <p className="font-mono text-xs break-words text-muted-foreground">{modelError}</p>
                  <Button variant="secondary" size="sm" onClick={retry}>
                    <RotateCcw />
                    Try again
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3 px-6 text-center text-muted-foreground">
                  <Film className="size-8" />
                  <p className="text-sm">Your swapped video will play here.</p>
                </div>
              )}
            </div>

            {(status === "done" || status === "error") && (
              <div className="flex flex-wrap items-center justify-end gap-2">
                <Button variant="secondary" size="sm" onClick={reset}>
                  <RotateCcw />
                  Reset
                </Button>
                {status === "done" && resultUrl && (
                  <Button variant="outline" size="sm" asChild>
                    <a href={resultUrl} download={RESULT_FILENAME}>
                      <Download />
                      Download
                    </a>
                  </Button>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
