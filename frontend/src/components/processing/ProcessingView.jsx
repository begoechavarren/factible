import documentIcon from '@assets/images/document.png';
import magnifierIcon from '@assets/images/magnifier.png';
import shieldIcon from '@assets/images/shield.png';
import printerIcon from '@assets/images/printer.png';

const STEPS = [
  { key: 'transcript', label: 'Extracting transcript' },
  { key: 'claim', label: 'Analyzing claims' },
  { key: 'processing', label: 'Fact-checking' },
  { key: 'generating', label: 'Generating report' },
];

export function ProcessingView({ progress, currentMessage, currentStep }) {
  const normalizedStep = (currentStep || '').toLowerCase();

  const activeIcon = (() => {
    if (normalizedStep.includes('generating')) {
      return { src: printerIcon, alt: 'Generating report' };
    }
    if (normalizedStep.includes('processing')) {
      return { src: shieldIcon, alt: 'Fact-checking claim' };
    }
    if (normalizedStep.includes('claim')) {
      return { src: magnifierIcon, alt: 'Analyzing claims' };
    }
    return { src: documentIcon, alt: 'Extracting transcript' };
  })();

  return (
    <section className="flex min-h-[24vh] w-full max-w-2xl flex-col items-center justify-center gap-8 px-4 text-center">
      <img
        src={activeIcon.src}
        alt={activeIcon.alt}
        className="h-12 w-12 animate-pulse object-contain drop-shadow-[0_0_6px_rgba(75,15,192,0.25)] md:h-14 md:w-14"
      />

      <div className="w-full max-w-md -mt-1 md:-mt-2">
        <div className="mb-3 flex items-center justify-between gap-2">
          <p className="pixel-text text-sm text-primary truncate">
            {currentMessage || 'Working on it...'}
          </p>
          <p className="pixel-text text-sm text-accent flex-shrink-0">{progress}%</p>
        </div>
        <div className="h-3 w-full rounded-full border-2 border-primary bg-background/50">
          <div
            className="h-full rounded-full bg-primary transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="mt-1 flex flex-col gap-2">
        {STEPS.map((step, index) => (
          <Step
            key={step.key}
            label={step.label}
            active={currentStep.includes(step.key)}
            completed={progress >= (index + 1) * 25}
          />
        ))}
      </div>
    </section>
  );
}

function Step({ label, active, completed }) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`h-2 w-2 rounded-full transition-colors ${
          completed ? 'bg-accent' : active ? 'bg-primary animate-pulse' : 'bg-gray-300'
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
