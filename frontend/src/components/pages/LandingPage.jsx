import { useState } from 'react';
import PageLayout from '@components/layout/PageLayout';
import Header from '@components/layout/Header';
import Footer from '@components/layout/Footer';
import SearchBar from '@components/search/SearchBar';
import FeatureList from '@components/features/FeatureList';

function LandingPage({ onFactCheck }) {
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (url) => {
    setLoading(true);
    // Pass to parent App component which will handle the fact-check flow
    onFactCheck(url);
  };

  return (
    <PageLayout>
      <Header />

      <main className="flex flex-1 items-center justify-center px-4 md:px-10 lg:px-12 py-16 md:py-20">
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
