import Logo from '@components/ui/Logo';

function Header({ onHome }) {
  const handleHome = () => {
    if (onHome) {
      onHome();
    }
  };

  return (
    <header className="absolute left-6 right-6 top-6 z-20 flex items-start justify-between gap-4 md:left-12 md:right-12 md:top-10">
      <button
        type="button"
        onClick={handleHome}
        className="group flex flex-col items-start gap-1 rounded-xl border border-transparent bg-transparent p-1 text-left focus:outline-none focus-visible:ring-0 cursor-pointer"
      >
        <span className="flex items-center gap-4">
          <Logo size="md" className="transition-transform duration-300 group-hover:scale-105" />
          <span className="pixel-text text-5xl leading-none text-primary transition-colors group-hover:text-accent md:text-6xl">
            factible
          </span>
        </span>
        <span className="pixel-text text-xs text-accent opacity-90 transition-colors group-hover:text-accent md:text-sm">
          YouTube Fact Checking app
        </span>
      </button>
    </header>
  );
}

export default Header;
