import { useEffect } from 'react';
import { useYouTubeUrl } from '@hooks/useYouTubeUrl';
import SearchInput from './SearchInput';
import SearchButton from './SearchButton';

function SearchBar({ onSubmit, disabled = false, loading = false, compact = false, resetSignal = 0 }) {
  const { url, isValid, error, setUrl, reset: resetUrl } = useYouTubeUrl();

  const handleSubmit = () => {
    if (isValid && onSubmit) {
      onSubmit(url);
    }
  };

  useEffect(() => {
    resetUrl();
  }, [resetSignal, resetUrl]);

  const containerClasses = [
    'transition-all duration-500 ease-out',
    'w-full',
    compact ? 'max-w-2xl scale-95' : 'max-w-2xl',
  ].join(' ');

  const frameClasses = [
    'relative flex items-center gap-2 rounded-full border',
    'bg-[rgba(255,255,255,0.95)]',
    'shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_4px_10px_rgba(0,0,0,0.06)]',
    'transition-smooth',
    'hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_6px_15px_rgba(0,0,0,0.1)]',
    'focus-within:border-[rgba(233,78,58,0.4)]',
    compact ? 'py-2 pl-5 pr-2 md:py-2 md:pl-5 md:pr-2' : 'py-2 pl-5 pr-2 md:py-2.5 md:pl-6 md:pr-2.5',
    compact ? 'border-[rgba(233,78,58,0.25)]' : 'border-[rgba(233,78,58,0.2)]',
  ].join(' ');

  return (
    <div className={containerClasses}>
      <div className={frameClasses}>
        <SearchInput
          value={url}
          onChange={setUrl}
          onSubmit={handleSubmit}
          isValid={isValid}
          error={error}
          disabled={disabled || loading}
        />
        <SearchButton
          onClick={handleSubmit}
          disabled={!isValid || disabled || loading}
          loading={loading}
        />
      </div>
    </div>
  );
}

export default SearchBar;
