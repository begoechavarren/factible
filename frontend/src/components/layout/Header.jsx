import Logo from '@components/ui/Logo';

function Header() {
  return (
    <header className="absolute left-8 right-8 top-8 z-10 flex items-start justify-between gap-4 md:left-12 md:top-12">
      <div className="flex flex-col gap-1 animate-fade-in">
        <div className="flex items-center gap-4">
          <Logo size="md" />
          <h1 className="pixel-text text-5xl text-primary md:text-6xl leading-none">
            factible
          </h1>
        </div>
        <p className="pixel-text text-xs text-accent opacity-90 md:text-sm text-center">
          YouTube Fact Checking app
        </p>
      </div>
    </header>
  );
}

export default Header;
