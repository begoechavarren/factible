import { useState, useCallback, useRef } from 'react';

/**
 * Hook to handle SSE streaming fact-check requests
 */
export function useFactCheck() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [currentMessage, setCurrentMessage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const eventSourceRef = useRef(null);

  const startFactCheck = useCallback(async (videoUrl, options = {}) => {
    // Reset state
    setIsProcessing(true);
    setProgress(0);
    setCurrentStep('');
    setCurrentMessage('');
    setResult(null);
    setError(null);

    const {
      maxClaims = 1,
      maxQueriesPerClaim = 1,
      maxResultsPerQuery = 1,
    } = options;

    try {
      const response = await fetch('http://localhost:8000/api/v1/fact-check/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_url: videoUrl,
          max_claims: maxClaims,
          max_queries_per_claim: maxQueriesPerClaim,
          max_results_per_query: maxResultsPerQuery,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          buffer += decoder.decode();
        } else {
          buffer += decoder.decode(value, { stream: true });
        }

        let separatorIndex;
        while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
          const rawEvent = buffer.slice(0, separatorIndex);
          buffer = buffer.slice(separatorIndex + 2);

          const dataLine = rawEvent.split('\n').find((line) => line.startsWith('data: '));
          if (!dataLine) {
            continue;
          }

          try {
            const data = JSON.parse(dataLine.slice(6));

            setCurrentStep(data.step);
            setCurrentMessage(data.message);
            setProgress(data.progress);

            if (data.step === 'complete') {
              setResult(data.data?.result);
              setIsProcessing(false);
            } else if (data.step === 'error') {
              setError(data.data?.error || data.message);
              setIsProcessing(false);
            }
          } catch (parseErr) {
            console.warn('Failed to parse SSE payload', parseErr);
          }
        }

        if (done) {
          break;
        }
      }
    } catch (err) {
      setError(err.message);
      setIsProcessing(false);
    }
  }, []);

  const cancel = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsProcessing(false);
  }, []);

  const reset = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsProcessing(false);
    setProgress(0);
    setCurrentStep('');
    setCurrentMessage('');
    setResult(null);
    setError(null);
  }, []);

  return {
    isProcessing,
    progress,
    currentStep,
    currentMessage,
    result,
    error,
    startFactCheck,
    cancel,
    reset,
  };
}
