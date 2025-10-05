# factible Frontend

Modern, responsive React application for fact-checking YouTube videos.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- npm, yarn, or pnpm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

### Development

```bash
# Run dev server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Format code
npm run format
```

## 📁 Project Structure

```
src/
├── components/        # React components
│   ├── features/     # Feature-specific components
│   ├── layout/       # Layout components (Header, Footer)
│   ├── pages/        # Page components
│   ├── search/       # Search-related components
│   └── ui/           # Reusable UI components
├── hooks/            # Custom React hooks
├── styles/           # Global styles
└── assets/           # Static assets (images, fonts)
```

## 🎨 Design System

### Colors
- **Background**: `#f3f1de` - Warm beige
- **Primary**: `#3D348B` - Deep blue
- **Accent**: `#E94E3A` - Orangy-red

### Fonts
- **Pixel**: VT323 (retro aesthetic)
- **Clean**: Inter (modern readability)

## 🧩 Key Components

- **LandingPage**: Main entry point
- **SearchBar**: YouTube URL input with validation
- **FeatureList**: Display app features
- **Header/Footer**: Consistent layout elements

## 🔧 Tech Stack

- **React 18.3** - UI library
- **Vite 6.0** - Build tool
- **Tailwind CSS 3.4** - Styling
- **ESLint + Prettier** - Code quality

## 📖 Documentation

See [FRD.md](./FRD.md) for complete frontend requirements and architecture.

## 🤝 Contributing

1. Follow the code style in `.prettierrc` and `.eslintrc.cjs`
2. Use path aliases (`@components`, `@hooks`, etc.)
3. Write accessible, semantic HTML
4. Test on mobile and desktop

## 📄 License

[To be determined]

## 👩‍💻 Author

Built by [@begoechavarren](https://github.com/begoechavarren)
