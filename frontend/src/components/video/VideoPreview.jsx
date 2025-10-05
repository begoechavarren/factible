function SkeletonCard() {
  return (
    <div className="mt-2 w-full max-w-xl animate-pulse overflow-hidden rounded-2xl border-2 border-primary/15 bg-white/60 p-3 shadow-inner backdrop-blur md:max-w-2xl">
      <div className="h-28 w-full rounded-xl bg-primary/10" />
      <div className="mt-3 h-4 w-3/4 rounded bg-primary/10" />
      <div className="mt-2 h-3 w-1/2 rounded bg-primary/10" />
    </div>
  );
}

export function VideoPreview({ url, metadata, loading }) {
  if (!url || (!metadata && !loading)) {
    return null;
  }

  if (loading) {
    return <SkeletonCard />;
  }

  return (
    <article className="mt-2 w-full max-w-xl overflow-hidden rounded-2xl border-2 border-primary/15 bg-white/85 shadow-lg backdrop-blur md:max-w-2xl">
      <div className="flex items-center gap-4 p-4">
        <div className="relative h-16 w-28 flex-shrink-0 overflow-hidden rounded-xl bg-primary/10">
          {metadata?.thumbnail ? (
            <img
              src={metadata.thumbnail}
              alt={metadata.title || 'YouTube thumbnail'}
              className="h-full w-full object-cover"
              loading="lazy"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center">
              <span className="pixel-text text-xs text-primary">Preview</span>
            </div>
          )}
        </div>
        <div className="flex-1">
          <p className="pixel-text text-[11px] uppercase tracking-[0.35em] text-accent/70">
            Analyzing
          </p>
          <h3 className="text-sm font-semibold leading-snug text-gray-800">
            {metadata?.title}
          </h3>
          {metadata?.author && (
            <p className="text-xs text-gray-500">by {metadata.author}</p>
          )}
        </div>
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="pixel-text text-xs uppercase text-primary hover:underline"
        >
          Open
        </a>
      </div>
    </article>
  );
}
