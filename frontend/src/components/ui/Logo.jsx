import logoImage from '@assets/images/logo.png';

const SIZE_CLASSES = {
  sm: 'w-12 h-12',
  md: 'w-16 h-16',
  lg: 'w-20 h-20',
  xl: 'w-24 h-24',
};

function Logo({ size = 'md', className = '' }) {
  const sizeClass = SIZE_CLASSES[size] || SIZE_CLASSES.md;

  return (
    <img
      src={logoImage}
      alt="factible logo"
      className={`${sizeClass} ${className} object-contain transition-smooth hover:scale-110`}
      style={{ filter: 'none' }}
    />
  );
}

export default Logo;
