import DotPattern from '@components/ui/DotPattern';

function PageLayout({ children }) {
  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-background">
      <DotPattern />
      {children}
    </div>
  );
}

export default PageLayout;
