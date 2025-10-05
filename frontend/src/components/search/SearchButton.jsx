function SearchButton({ onClick, disabled = false, loading = false }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={`
        grid h-10 w-10 place-items-center rounded-full
        border border-[rgba(233,78,58,0.3)] bg-primary text-white
        shadow-[inset_0_0_0_1px_rgba(255,255,255,0.3),0_4px_10px_rgba(61,52,139,0.25)]
        transition-smooth
        hover:scale-105 hover:shadow-lg
        active:scale-95
        disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100
        focus-ring
        md:h-11 md:w-11
      `}
      aria-label={loading ? 'Processing' : 'Start fact-checking'}
    >
      <span className={`pixel-text text-2xl leading-none ${loading ? 'animate-pulse' : ''}`}>
        {loading ? '⋯' : '→'}
      </span>
    </button>
  );
}

export default SearchButton;
