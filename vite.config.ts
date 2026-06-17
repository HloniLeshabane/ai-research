import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Pyodide assets load from the pinned CDN at runtime inside the worker, so
// there's nothing to externalize. ES-format workers are required (pyodide.mjs
// is an ES module). dedupe + optimizeDeps.include force a single React instance
// so @uiw/react-codemirror can't pull in a second copy (which causes React's
// "Invalid hook call" crash).
export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },
  resolve: { dedupe: ['react', 'react-dom'] },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react/jsx-runtime', '@uiw/react-codemirror'],
  },
});
