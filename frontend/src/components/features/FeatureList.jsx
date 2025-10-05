import FeaturePill from '@components/ui/FeaturePill';

const FEATURES = [
  'AI‑Powered Analysis',
  'Source Verification',
  'Real‑time Results',
];

function FeatureList() {
  return (
    <div className="mt-6 flex items-center justify-center gap-3 md:gap-4 animate-fade-in">
      {FEATURES.map((feature, index) => (
        <FeaturePill
          key={feature}
          className="animate-slide-up"
          style={{ animationDelay: `${index * 100}ms` }}
        >
          {feature}
        </FeaturePill>
      ))}
    </div>
  );
}

export default FeatureList;
