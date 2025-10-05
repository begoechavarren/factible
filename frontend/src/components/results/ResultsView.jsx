import Logo from '@components/ui/Logo';

export function ResultsView({ result, onReset }) {
  if (!result || !result.claim_reports) {
    return null;
  }

  const processedClaims = result.claim_reports.length;
  const totalSources = result.claim_reports.reduce(
    (count, report) => count + (report.total_sources || 0),
    0,
  );

  return (
    <section className="mx-auto w-full max-w-5xl space-y-8 animate-fade-in">
      <div className="flex flex-col gap-6 rounded-3xl border-2 border-primary/25 bg-white/90 p-8 shadow-xl backdrop-blur md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-4">
          <Logo size="sm" className="hidden md:block" />
          <div>
            <p className="pixel-text text-xs uppercase tracking-[0.4em] text-accent/70">
              Fact-check complete
            </p>
            <h2 className="pixel-text text-2xl text-primary md:text-3xl">Analysis summary</h2>
            <p className="mt-2 text-sm text-gray-600">
              {processedClaims} claim{processedClaims === 1 ? '' : 's'} processed using {totalSources}{' '}
              evidence source{totalSources === 1 ? '' : 's'}.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <SummaryChip label="Claims" value={processedClaims} />
          <SummaryChip label="Evidence sources" value={totalSources} />
          <button
            onClick={onReset}
            className="pixel-text rounded-xl border-2 border-primary px-5 py-2 text-sm text-primary transition-colors hover:bg-primary hover:text-background"
          >
            New check
          </button>
        </div>
      </div>

      <div className="space-y-8">
        {result.claim_reports.map((report, index) => (
          <ClaimCard key={report.claim_text ?? index} report={report} index={index + 1} />
        ))}
      </div>
    </section>
  );
}

function SummaryChip({ label, value }) {
  return (
    <div className="rounded-xl border border-primary/20 bg-primary/5 px-4 py-2 text-left">
      <p className="pixel-text text-xs uppercase tracking-widest text-primary/70">{label}</p>
      <p className="pixel-text text-lg text-primary">{value}</p>
    </div>
  );
}

function ClaimCard({ report, index }) {
  const stanceColors = {
    supports: 'border-green-400/60 bg-green-50/70 text-green-700',
    refutes: 'border-red-400/60 bg-red-50/70 text-red-700',
    mixed: 'border-yellow-400/70 bg-yellow-50/70 text-yellow-700',
    unclear: 'border-slate-300 bg-slate-50 text-slate-600',
  };

  const stanceStyle = stanceColors[report.overall_stance] || stanceColors.unclear;

  return (
    <article className="rounded-3xl border-2 border-primary/20 bg-white/95 p-6 shadow-lg backdrop-blur transition-all duration-300 hover:-translate-y-1 hover:shadow-xl">
      <div className="mb-5 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="flex-1">
          <p className="pixel-text mb-2 text-xs uppercase tracking-widest text-accent">Claim {index}</p>
          <p className="text-base leading-relaxed text-gray-800">{report.claim_text}</p>
          <p className="pixel-text mt-2 text-[11px] uppercase text-gray-400">
            Category: {report.claim_category} · Extraction confidence {Math.round(report.claim_confidence * 100)}%
          </p>
        </div>
        <div className={`pixel-text self-start rounded-xl border-2 px-4 py-2 text-xs uppercase ${stanceStyle}`}>
          {report.overall_stance}
        </div>
      </div>

      <div className="mb-6 rounded-2xl border border-primary/10 bg-primary/5 p-4">
        <p className="pixel-text mb-2 text-xs uppercase tracking-widest text-primary/80">Verdict</p>
        <p className="text-sm leading-relaxed text-gray-700">{report.verdict_summary}</p>
        <p className="pixel-text mt-3 text-xs text-gray-500">
          Confidence level: {report.verdict_confidence}
        </p>
      </div>

      <EvidenceGroups evidenceByStance={report.evidence_by_stance} />
    </article>
  );
}

function EvidenceGroups({ evidenceByStance }) {
  const entries = Object.entries(evidenceByStance || {});

  if (!entries.length) {
    return (
      <p className="pixel-text text-xs text-gray-400">No online evidence retrieved for this claim yet.</p>
    );
  }

  return (
    <div className="grid gap-5 md:grid-cols-2">
      {entries.map(([stance, sources]) => (
        <div key={stance} className="rounded-2xl border border-primary/10 bg-primary/5 p-4">
          <p className="pixel-text mb-3 text-xs uppercase tracking-widest text-gray-500">
            {stance} ({sources.length} source{sources.length > 1 ? 's' : ''})
          </p>
          <div className="space-y-3">
            {sources.map((source, index) => (
              <SourceCard key={`${stance}-${index}`} source={source} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function SourceCard({ source }) {
  const reliabilityColors = {
    high: 'text-green-600',
    medium: 'text-yellow-600',
    low: 'text-red-600',
    unknown: 'text-gray-500',
  };

  const reliability = source.reliability?.rating ?? 'unknown';
  const reliabilityColor = reliabilityColors[reliability] || reliabilityColors.unknown;

  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group block rounded-xl border border-primary/10 bg-white/90 p-4 transition-all duration-300 hover:border-primary/40 hover:shadow-md"
    >
      <div className="mb-2 flex items-start justify-between gap-4">
        <p className="flex-1 text-sm font-medium text-gray-800 group-hover:text-primary">{source.title}</p>
        <p className={`pixel-text text-xs uppercase ${reliabilityColor}`}>{reliability}</p>
      </div>
      {source.evidence_summary && (
        <p className="mb-2 text-xs text-gray-600">{source.evidence_summary}</p>
      )}
      <p className="pixel-text text-xs text-gray-400 transition-colors group-hover:text-primary">
        {new URL(source.url).hostname}
      </p>
    </a>
  );
}
