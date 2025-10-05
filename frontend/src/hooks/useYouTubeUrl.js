import { useState, useCallback } from 'react';

const YOUTUBE_URL_PATTERNS = [
  /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]+/,
  /^https?:\/\/(www\.)?youtu\.be\/[\w-]+/,
  /^https?:\/\/(www\.)?youtube\.com\/embed\/[\w-]+/,
  /^https?:\/\/(www\.)?youtube\.com\/v\/[\w-]+/,
];

export function useYouTubeUrl() {
  const [url, setUrl] = useState('');
  const [isValid, setIsValid] = useState(false);
  const [error, setError] = useState('');

  const validateUrl = useCallback((input) => {
    if (!input.trim()) {
      setIsValid(false);
      setError('');
      return false;
    }

    const valid = YOUTUBE_URL_PATTERNS.some((pattern) => pattern.test(input));
    setIsValid(valid);
    setError(valid ? '' : 'Please enter a valid YouTube URL');
    return valid;
  }, []);

  const handleChange = useCallback(
    (newUrl) => {
      setUrl(newUrl);
      validateUrl(newUrl);
    },
    [validateUrl]
  );

  const reset = useCallback(() => {
    setUrl('');
    setIsValid(false);
    setError('');
  }, []);

  return {
    url,
    isValid,
    error,
    setUrl: handleChange,
    reset,
  };
}
