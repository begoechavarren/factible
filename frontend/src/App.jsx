import { useMemo, useRef, useState } from 'react';
import Header from '@components/layout/Header';
import Footer from '@components/layout/Footer';
import PageLayout from '@components/layout/PageLayout';
import SearchBar from '@components/search/SearchBar';
import FeatureList from '@components/features/FeatureList';
import { ProcessingView } from '@components/processing/ProcessingView';
import { InteractiveResultsView } from '@components/results/InteractiveResultsView';
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
    const classes = [
      'flex flex-1 w-full flex-col items-center px-4 md:px-10 lg:px-12 pb-16 md:pb-20',
      'transition-all duration-500 ease-out',
    ];
    if (hasActiveRun) {
      classes.push('justify-start pt-3 md:pt-4 lg:pt-5');
    } else {
      classes.push('justify-start pt-16 md:pt-20');
    }
    return classes.join(' ');
  }, [hasActiveRun]);

  const contentWrapperClass = useMemo(
    () =>
      hasActiveRun
        ? 'mx-auto w-full max-w-5xl flex flex-col items-center gap-6 md:gap-7 lg:gap-8 transition-all duration-500 ease-out'
        : 'mx-auto w-full max-w-5xl flex flex-col items-center gap-9 md:gap-12 transition-all duration-500 ease-out',
    [hasActiveRun],
  );

  const heroClassName = useMemo(
    () =>
      [
        'w-full max-w-2xl mx-auto',
        hasActiveRun ? 'flex flex-col items-center gap-4 md:gap-5' : 'flex flex-col items-center gap-6 md:gap-7',
        'transform transition-all duration-500 ease-out',
        hasActiveRun ? 'mt-2 md:mt-3 -translate-y-2 md:-translate-y-3' : 'mt-12 md:mt-14 translate-y-0',
      ].join(' '),
    [hasActiveRun],
  );

  return (
    <PageLayout>
      <Header onHome={handleReset} compact={hasActiveRun} />
      <main className={mainClassName}>
        <div className={contentWrapperClass}>
          {!result && (
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
          )}

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
            <div className="w-full flex justify-center -mt-3 md:-mt-6 transition-all duration-500 ease-out">
              <ProcessingView
                progress={progress}
                currentMessage={currentMessage}
                currentStep={currentStep}
              />
            </div>
          )}

          {result && (
            <InteractiveResultsView result={result} onReset={handleReset} />
          )}
        </div>
      </main>
      <Footer />
    </PageLayout>
  );
}

export default App;
