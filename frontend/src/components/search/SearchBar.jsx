import { useYouTubeUrl } from '@hooks/useYouTubeUrl';
import SearchInput from './SearchInput';
import SearchButton from './SearchButton';

function SearchBar({ onSubmit, disabled = false, loading = false }) {
  const { url, isValid, error, setUrl } = useYouTubeUrl();

  const handleSubmit = () => {
    if (isValid && onSubmit) {
      onSubmit(url);
    }
  };

  return (
    <div className="w-full max-w-2xl animate-slide-up">
      <div
        className="
          relative flex items-center gap-2 rounded-full border
          border-[rgba(233,78,58,0.2)] bg-[rgba(255,255,255,0.95)]
          py-2 pl-5 pr-2 md:py-2.5 md:pl-6 md:pr-2.5
          shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_4px_10px_rgba(0,0,0,0.06)]
          transition-smooth
          hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_6px_15px_rgba(0,0,0,0.1)]
          focus-within:border-[rgba(233,78,58,0.4)]
        "
      >
        <SearchInput
          value={url}
          onChange={setUrl}
          onSubmit={handleSubmit}
          isValid={isValid}
          error={error}
          disabled={disabled || loading}
        />
        <SearchButton onClick={handleSubmit} disabled={!isValid || disabled} loading={loading} />
      </div>
    </div>
  );
}

export default SearchBar;
