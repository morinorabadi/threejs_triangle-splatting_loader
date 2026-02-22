import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss()],
  server: {
    port: 8001,
    host: true,
  },
  build: {
    outDir: 'dist',
    minify: 'esbuild',
  },
})
