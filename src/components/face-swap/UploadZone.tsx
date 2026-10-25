import { useRef, useState, type DragEvent, type ReactNode } from "react";
import { X } from "lucide-react";

import { cn } from "@/lib/utils";

type UploadZoneProps = {
  step: string;
  label: string;
  hint: string;
  accept: string;
  icon: ReactNode;
  file: File | null;
  previewUrl: string | null;
  kind: "image" | "video";
  onFile: (file: File) => void;
  onClear: () => void;
  validate: (file: File) => string | null;
};

export function UploadZone({
  step,
  label,
  hint,
  accept,
  icon,
  file,
  previewUrl,
  kind,
  onFile,
  onClear,
  validate,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handle = (f: File | undefined) => {
    if (!f) return;
    const problem = validate(f);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    onFile(f);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    handle(e.dataTransfer.files?.[0]);
  };

  return (
    <section className="flex flex-col gap-3">
      <header className="flex items-baseline justify-between">
        <h2 className="text-sm font-semibold tracking-tight">
          <span className="mr-2 font-mono text-xs text-primary">{step}</span>
          {label}
        </h2>
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          {hint}
        </span>
      </header>

      <div
        role="button"
        tabIndex={0}
        aria-label={`Upload ${label}`}
        onClick={() => !file && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!file && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          "relative flex aspect-[4/3] items-center justify-center overflow-hidden rounded-xl border border-dashed bg-dropzone transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          !file && "cursor-pointer hover:border-primary/60 hover:bg-surface",
          dragging && "border-primary bg-surface",
          file && "border-solid",
        )}
      >
        {file && previewUrl ? (
          <>
            {kind === "image" ? (
              <img src={previewUrl} alt="Source face preview" className="h-full w-full object-contain" />
            ) : (
              <video src={previewUrl} className="h-full w-full object-contain" muted loop autoPlay playsInline />
            )}
            <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-2 bg-background/80 px-3 py-2 backdrop-blur">
              <span className="truncate font-mono text-xs text-muted-foreground">{file.name}</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onClear();
                  if (inputRef.current) inputRef.current.value = "";
                }}
                className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                aria-label={`Remove ${label}`}
              >
                <X className="size-4" />
              </button>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center gap-3 px-6 text-center">
            <div className="flex size-12 items-center justify-center rounded-full bg-surface-raised text-primary">
              {icon}
            </div>
            <div>
              <p className="text-sm font-medium">Drop file here or click to browse</p>
              <p className="mt-1 font-mono text-xs text-muted-foreground">{accept.replaceAll(",", " · ")}</p>
            </div>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => handle(e.target.files?.[0])}
        />
      </div>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </section>
  );
}
