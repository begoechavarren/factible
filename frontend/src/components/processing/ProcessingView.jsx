import Logo from '@components/ui/Logo';

export function ProcessingView({ progress, currentMessage, currentStep }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-8 px-4">
      {/* Logo with pulsing animation */}
      <div className="animate-pulse">
        <Logo size="lg" />
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-md">
        <div className="mb-3 flex items-center justify-between">
          <p className="pixel-text text-sm text-primary">{currentMessage}</p>
          <p className="pixel-text text-sm text-accent">{progress}%</p>
        </div>

        <div className="h-3 w-full rounded-full border-2 border-primary bg-background/50">
          <div
            className="h-full rounded-full bg-primary transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Processing steps indicator */}
      <div className="flex flex-col gap-2">
        <Step
          label="Extracting transcript"
          active={currentStep.includes('transcript')}
          completed={progress > 15}
        />
        <Step
          label="Analyzing claims"
          active={currentStep.includes('claim')}
          completed={progress > 35}
        />
        <Step
          label="Fact-checking"
          active={currentStep.includes('processing')}
          completed={progress > 85}
        />
        <Step
          label="Generating report"
          active={currentStep.includes('generating')}
          completed={progress === 100}
        />
      </div>
    </div>
  );
}

function Step({ label, active, completed }) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`h-2 w-2 rounded-full transition-colors ${
          completed
            ? 'bg-accent'
            : active
              ? 'bg-primary animate-pulse'
              : 'bg-gray-300'
        }`}
      />
      <p
        className={`pixel-text text-sm transition-colors ${
          completed || active ? 'text-primary' : 'text-gray-400'
        }`}
      >
        {label}
      </p>
    </div>
  );
}
