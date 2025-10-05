import { useState } from 'react';
import LandingPage from '@components/pages/LandingPage';
import { ProcessingView } from '@components/processing/ProcessingView';
import { ResultsView } from '@components/results/ResultsView';
import { useFactCheck } from '@hooks/useFactCheck';

function App() {
  const [videoUrl, setVideoUrl] = useState('');
  const {
    isProcessing,
    progress,
    currentStep,
    currentMessage,
    result,
    error,
    startFactCheck,
    reset,
  } = useFactCheck();

  const handleFactCheck = (url) => {
    setVideoUrl(url);
    startFactCheck(url);
  };

  const handleReset = () => {
    reset();
    setVideoUrl('');
  };

  const handleRetry = () => {
    if (!videoUrl) {
      handleReset();
      return;
    }

    reset();
    startFactCheck(videoUrl);
  };

  // Show results view if we have results
  if (result) {
    return <ResultsView result={result} onReset={handleReset} />;
  }

  // Show processing view while processing
  if (isProcessing) {
    return (
      <ProcessingView
        progress={progress}
        currentMessage={currentMessage}
        currentStep={currentStep}
      />
    );
  }

  // Show error if there's an error
  if (error) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-4">
        <p className="pixel-text text-accent">Error: {error}</p>
        <button
          onClick={handleRetry}
          className="pixel-text rounded-lg border-2 border-primary px-6 py-2 text-primary transition-colors hover:bg-primary hover:text-background"
        >
          Try Again
        </button>
      </div>
    );
  }

  // Show landing page by default
  return <LandingPage onFactCheck={handleFactCheck} />;
}

export default App;
