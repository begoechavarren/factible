function Footer() {
  return (
    <footer className="absolute bottom-4 left-0 right-0 flex justify-center px-4 md:bottom-6">
      <span className="pixel-text text-center text-xs text-accent opacity-70 md:text-sm">
        👩‍💻 Built by{' '}
        <a
          href="https://github.com/begoechavarren"
          target="_blank"
          rel="noopener noreferrer"
          className="underline transition-smooth hover:opacity-100 hover:text-primary"
        >
          @begoechavarren
        </a>{' '}
        at{' '}
        <a
          href="https://github.com/begoechavarren/factible"
          target="_blank"
          rel="noopener noreferrer"
          className="underline transition-smooth hover:opacity-100 hover:text-primary"
        >
          factible
        </a>
      </span>
    </footer>
  );
}

export default Footer;
