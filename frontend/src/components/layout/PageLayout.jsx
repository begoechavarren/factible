import DotPattern from '@components/ui/DotPattern';

function PageLayout({ children }) {
  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-background">
      <DotPattern />
      <div className="relative z-10 flex min-h-screen flex-col">
        {children}
      </div>
    </div>
  );
}

export default PageLayout;
