import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Film, ScanFace, Sparkles, Download, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
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

function useObjectUrl(file: File | null) {
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

type Status = "idle" | "processing" | "done";

function Index() {
  const [face, setFace] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const faceUrl = useObjectUrl(face);
  const videoUrl = useObjectUrl(video);

  useEffect(() => () => {
    if (timer.current) clearInterval(timer.current);
  }, []);

  const reset = () => {
    if (timer.current) clearInterval(timer.current);
    setStatus("idle");
    setProgress(0);
  };

  // Placeholder run: no processing engine is connected yet, so this only
  // animates progress and then shows the target clip in the result player.
  const start = () => {
    if (!face || !video) return;
    setStatus("processing");
    setProgress(0);
    timer.current = setInterval(() => {
      setProgress((p) => {
        const next = Math.min(100, p + 4);
        if (next >= 100 && timer.current) {
          clearInterval(timer.current);
          setStatus("done");
        }
        return next;
      });
    }, 80);
  };

  const ready = Boolean(face && video) && status !== "processing";

  return (
    <main className="min-h-screen bg-grid">
      <div className="mx-auto flex max-w-6xl flex-col gap-10 px-5 py-10 md:py-16">
        <header className="flex flex-col gap-3">
          <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-primary">
            <span className="size-1.5 rounded-full bg-primary" />
            Swapframe · runs locally
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

            <Button variant="action" size="xl" disabled={!ready} onClick={start} className="w-full">
              <Sparkles />
              {status === "processing" ? "Swapping…" : "Start Face Swap"}
            </Button>
            <p className="-mt-3 text-center text-xs text-muted-foreground">
              Only use faces and footage you have permission to use.
            </p>
          </div>

          <section className="flex flex-col gap-3">
            <header className="flex items-baseline justify-between">
              <h2 className="text-sm font-semibold tracking-tight">
                <span className="mr-2 font-mono text-xs text-primary">03</span>
                Result
              </h2>
              <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                {status === "idle" && "Waiting"}
                {status === "processing" && `${progress}%`}
                {status === "done" && "Ready"}
              </span>
            </header>

            <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-2xl border bg-surface">
              {status === "done" && videoUrl ? (
                <video key={videoUrl} src={videoUrl} controls autoPlay playsInline className="h-full w-full bg-background object-contain" />
              ) : status === "processing" ? (
                <div className="flex w-full max-w-xs flex-col items-center gap-4 px-6 text-center">
                  <ScanFace className="size-8 animate-pulse text-primary" />
                  <Progress value={progress} className="h-1.5 bg-muted [&>div]:bg-primary" />
                  <p className="font-mono text-xs text-muted-foreground">Processing frames…</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3 px-6 text-center text-muted-foreground">
                  <Film className="size-8" />
                  <p className="text-sm">Your swapped video will play here.</p>
                </div>
              )}
            </div>

            {status === "done" && (
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs text-muted-foreground">
                  Preview only — the swap engine isn't connected yet, so this shows your original clip.
                </p>
                <div className="flex gap-2">
                  <Button variant="secondary" size="sm" onClick={reset}>
                    <RotateCcw />
                    Reset
                  </Button>
                  {videoUrl && video && (
                    <Button variant="outline" size="sm" asChild>
                      <a href={videoUrl} download={`swapped-${video.name}`}>
                        <Download />
                        Download
                      </a>
                    </Button>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
