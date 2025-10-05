import { useRef, useEffect } from 'react';

function SearchInput({ value, onChange, onSubmit, isValid, error, disabled = false }) {
  const inputRef = useRef(null);

  useEffect(() => {
    // Auto-focus input on mount
    inputRef.current?.focus();
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && isValid && onSubmit) {
      onSubmit();
    }
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="url"
        inputMode="url"
        placeholder="Paste YouTube link here…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className={`
          themed-input clean-text w-full flex-1 bg-transparent px-1 py-1.5
          text-sm text-accent caret-accent
          transition-smooth focus:outline-none
          md:text-base
          ${disabled ? 'cursor-not-allowed opacity-50' : ''}
        `}
        aria-label="YouTube video URL"
        aria-invalid={!!error}
        aria-describedby={error ? 'url-error' : undefined}
      />
      {error && (
        <p
          id="url-error"
          className="mt-3 text-sm text-accent opacity-70 animate-fade-in md:text-base"
          role="alert"
        >
          {error}
        </p>
      )}
    </div>
  );
}

export default SearchInput;
