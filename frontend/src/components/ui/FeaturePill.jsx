function FeaturePill({ children, className = '' }) {
  return (
    <span
      className={`
        pixel-text rounded-full border border-pill-border bg-pill-bg
        px-4 py-2 text-xs text-accent
        transition-smooth hover:scale-105 hover:shadow-md
        md:px-5 md:py-2.5 md:text-sm
        ${className}
      `}
    >
      {children}
    </span>
  );
}

export default FeaturePill;
