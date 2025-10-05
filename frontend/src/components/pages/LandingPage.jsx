import { useState } from 'react';
import PageLayout from '@components/layout/PageLayout';
import Header from '@components/layout/Header';
import Footer from '@components/layout/Footer';
import SearchBar from '@components/search/SearchBar';
import FeatureList from '@components/features/FeatureList';

function LandingPage() {
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (url) => {
    console.log('Submitting URL:', url);
    setLoading(true);

    // TODO: Integrate with backend API
    // This will be replaced with actual API call in the future
    setTimeout(() => {
      setLoading(false);
      alert(`Ready to fact-check: ${url}\n\nBackend integration coming soon!`);
    }, 1500);
  };

  return (
    <PageLayout>
      <Header />

      <main className="flex min-h-screen items-center justify-center px-4">
        <div className="w-full max-w-2xl">
          {/* Screen reader only description */}
          <p className="sr-only">
            Factible helps you fact‑check YouTube videos quickly and reliably using AI-powered
            analysis.
          </p>

          <SearchBar onSubmit={handleSubmit} loading={loading} />

          <FeatureList />
        </div>
      </main>

      <Footer />
    </PageLayout>
  );
}

export default LandingPage;
