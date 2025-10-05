import { useMemo, useRef, useState } from 'react';
import Header from '@components/layout/Header';
import Footer from '@components/layout/Footer';
import PageLayout from '@components/layout/PageLayout';
import SearchBar from '@components/search/SearchBar';
import FeatureList from '@components/features/FeatureList';
import { ProcessingView } from '@components/processing/ProcessingView';
import { ResultsView } from '@components/results/ResultsView';
import { useFactCheck } from '@hooks/useFactCheck';
import { VideoPreview } from '@components/video/VideoPreview';

function App() {
  const [videoUrl, setVideoUrl] = useState('');
  const [videoMeta, setVideoMeta] = useState(null);
  const [isMetaLoading, setIsMetaLoading] = useState(false);
  const [resetToken, setResetToken] = useState(0);
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

  const latestUrlRef = useRef('');

  const hasActiveRun = useMemo(
    () => Boolean(isProcessing || result || error),
    [isProcessing, result, error],
  );

  const fetchVideoMetadata = async (url) => {
    if (!url) {
      return;
    }

    latestUrlRef.current = url;
    setIsMetaLoading(true);

    try {
      const endpoint = `https://www.youtube.com/oembed?url=${encodeURIComponent(url)}&format=json`;
      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error('Failed to fetch video metadata');
      }

      const data = await response.json();

      if (latestUrlRef.current !== url) {
        return;
      }

      setVideoMeta({
        title: data.title,
        author: data.author_name,
        thumbnail: data.thumbnail_url,
      });
    } catch (err) {
      console.warn('Unable to fetch YouTube metadata:', err);
      if (latestUrlRef.current === url) {
        setVideoMeta(null);
      }
    } finally {
      if (latestUrlRef.current === url) {
        setIsMetaLoading(false);
      }
    }
  };

  const handleFactCheck = (url) => {
    setVideoUrl(url);
    setVideoMeta(null);
    fetchVideoMetadata(url);
    startFactCheck(url);
  };

  const handleReset = () => {
    reset();
    setVideoUrl('');
    setVideoMeta(null);
    setIsMetaLoading(false);
    latestUrlRef.current = '';
    setResetToken((token) => token + 1);
  };

  const handleRetry = () => {
    if (!videoUrl) {
      handleReset();
      return;
    }

    reset();
    if (!videoMeta) {
      fetchVideoMetadata(videoUrl);
    }
    startFactCheck(videoUrl);
  };

  const mainClassName = useMemo(() => {
    const classes = ['relative z-10 flex min-h-screen w-full flex-col px-4 pb-20'];
    if (hasActiveRun) {
      classes.push('items-center justify-start pt-24 md:pt-32');
    } else {
      classes.push('items-center justify-center pt-16 md:pt-20');
    }
    return classes.join(' ');
  }, [hasActiveRun]);

  const contentWrapperClass = useMemo(
    () =>
      hasActiveRun
        ? 'w-full max-w-5xl flex flex-col items-center gap-10'
        : 'w-full max-w-5xl flex flex-col items-center gap-10',
    [hasActiveRun],
  );

  const heroClassName = useMemo(
    () =>
      [
        'transition-transform duration-500 ease-out',
        'w-full max-w-2xl mx-auto',
        'flex flex-col items-center gap-6',
        hasActiveRun ? '-translate-y-16 md:-translate-y-20' : 'translate-y-0',
      ].join(' '),
    [hasActiveRun],
  );

  return (
    <PageLayout>
      <Header onHome={handleReset} />
      <main className={mainClassName}>
        <div className={contentWrapperClass}>
          <section className={heroClassName}>
            <div className="w-full">
              <SearchBar
                onSubmit={handleFactCheck}
                loading={isProcessing}
                compact={hasActiveRun}
                disabled={isProcessing}
                resetSignal={resetToken}
              />
            </div>

            {(isMetaLoading || (videoMeta && videoUrl)) && (
              <div className="w-full">
                <VideoPreview
                  url={videoUrl}
                  metadata={videoMeta}
                  loading={isMetaLoading}
                />
              </div>
            )}

            {!hasActiveRun && <FeatureList />}
          </section>

          {error && (
            <div className="mx-auto w-full max-w-3xl rounded-3xl border-2 border-accent/40 bg-white/90 p-8 shadow-lg backdrop-blur">
              <p className="pixel-text mb-4 text-center text-accent">Error: {error}</p>
              <div className="flex justify-center">
                <button
                  onClick={handleRetry}
                  className="pixel-text rounded-lg border-2 border-primary px-6 py-2 text-primary transition-colors hover:bg-primary hover:text-background"
                >
                  Try Again
                </button>
              </div>
            </div>
          )}

          {isProcessing && !result && (
            <div className="-mt-16 w-full flex justify-center">
              <ProcessingView
                progress={progress}
                currentMessage={currentMessage}
                currentStep={currentStep}
              />
            </div>
          )}

          {result && (
            <ResultsView result={result} onReset={handleReset} />
          )}
        </div>
      </main>
      <Footer />
    </PageLayout>
  );
}

export default App;
