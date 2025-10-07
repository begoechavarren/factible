import Logo from '@components/ui/Logo';

function Header({ onHome, compact = false }) {
  const handleHome = () => {
    if (onHome) {
      onHome();
    }
  };

  const headerClasses = [
    'relative z-20 w-full',
    compact ? 'pt-4 md:pt-6' : 'pt-6 md:pt-10',
  ].join(' ');

  return (
    <header className={headerClasses}>
      <div className="flex w-full items-start justify-between px-4 md:px-10 lg:px-12">
        <button
          type="button"
          onClick={handleHome}
          className="group flex flex-col items-start gap-1 rounded-xl border border-transparent bg-transparent p-1 text-left focus:outline-none focus-visible:ring-0 cursor-pointer"
        >
          <span className="flex items-center gap-3 md:gap-4">
            <Logo size="md" className="transition-transform duration-300 group-hover:scale-105" />
            <span className={`pixel-text leading-none text-primary transition-colors group-hover:text-accent ${
              compact ? 'text-4xl md:text-5xl lg:text-6xl' : 'text-5xl md:text-5xl lg:text-6xl'
            }`}>
              factible
            </span>
          </span>
          <span className={`pixel-text text-accent opacity-90 transition-colors group-hover:text-accent ml-14 md:ml-14 ${
            compact ? 'text-xs md:text-sm' : 'text-sm md:text-sm'
          }`}>
            YouTube Fact Checking app
          </span>
        </button>
      </div>
    </header>
  );
}

export default Header;
