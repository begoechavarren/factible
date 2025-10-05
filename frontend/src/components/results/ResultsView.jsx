import Logo from '@components/ui/Logo';

export function ResultsView({ result, onReset }) {
  if (!result || !result.claim_reports) {
    return null;
  }

  return (
    <div className="min-h-screen px-4 py-12">
      <div className="mx-auto max-w-4xl">
        {/* Header */}
        <div className="mb-12 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Logo size="sm" />
            <h1 className="pixel-text text-2xl text-primary">Fact-Check Results</h1>
          </div>
          <button
            onClick={onReset}
            className="pixel-text rounded-lg border-2 border-primary bg-background px-4 py-2 text-sm text-primary transition-colors hover:bg-primary hover:text-background"
          >
            New Check
          </button>
        </div>

        {/* Claims */}
        <div className="space-y-8">
          {result.claim_reports.map((report, idx) => (
            <ClaimCard key={idx} report={report} index={idx + 1} />
          ))}
        </div>
      </div>
    </div>
  );
}

function ClaimCard({ report, index }) {
  const stanceColors = {
    supports: 'border-green-500 bg-green-50',
    refutes: 'border-red-500 bg-red-50',
    neutral: 'border-gray-400 bg-gray-50',
  };

  const stanceColor = stanceColors[report.overall_stance] || stanceColors.neutral;

  return (
    <div className="animate-fade-in rounded-lg border-2 border-primary bg-white p-6 shadow-sm">
      {/* Claim header */}
      <div className="mb-4 flex items-start justify-between gap-4">
        <div className="flex-1">
          <p className="pixel-text mb-2 text-xs text-accent">Claim {index}</p>
          <p className="text-base leading-relaxed text-gray-800">
            {report.claim?.text || 'No claim text available'}
          </p>
        </div>
        <div
          className={`pixel-text rounded-lg border-2 px-3 py-1 text-xs uppercase ${stanceColor}`}
        >
          {report.overall_stance}
        </div>
      </div>

      {/* Verdict */}
      <div className="mb-6 rounded-lg bg-background/50 p-4">
        <p className="pixel-text mb-1 text-xs text-primary">Summary</p>
        <p className="text-sm leading-relaxed text-gray-700">
          {report.verdict_summary}
        </p>
        <p className="pixel-text mt-2 text-xs text-gray-500">
          Confidence: {report.verdict_confidence}
        </p>
      </div>

      {/* Evidence by stance */}
      {Object.entries(report.evidence_by_stance || {}).map(([stance, sources]) => {
        if (!sources || sources.length === 0) return null;

        return (
          <div key={stance} className="mb-4">
            <p className="pixel-text mb-3 text-xs uppercase text-gray-600">
              {stance} ({sources.length} source{sources.length > 1 ? 's' : ''})
            </p>
            <div className="space-y-3">
              {sources.map((source, idx) => (
                <SourceCard key={idx} source={source} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SourceCard({ source }) {
  const reliabilityColors = {
    high: 'text-green-600',
    medium: 'text-yellow-600',
    low: 'text-red-600',
  };

  const reliabilityColor =
    reliabilityColors[source.reliability?.rating] || 'text-gray-600';

  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block rounded-lg border border-gray-200 bg-gray-50 p-4 transition-all hover:border-primary hover:shadow-md"
    >
      <div className="mb-2 flex items-start justify-between gap-4">
        <p className="flex-1 text-sm font-medium text-gray-800">{source.title}</p>
        <p className={`pixel-text text-xs uppercase ${reliabilityColor}`}>
          {source.reliability?.rating || 'unknown'}
        </p>
      </div>

      {source.evidence_summary && (
        <p className="mb-2 text-xs text-gray-600">{source.evidence_summary}</p>
      )}

      <p className="pixel-text text-xs text-gray-400 hover:text-primary">
        {new URL(source.url).hostname}
      </p>
    </a>
  );
}
