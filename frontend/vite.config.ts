import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    // First match wins: the training monitor hub (`run-frontend.sh --monitor`) before the dataset API
    proxy: { '/api/hosts': 'http://127.0.0.1:8011', '/api': 'http://127.0.0.1:8010' },
  },
  build: { rollupOptions: { input: ['index.html', 'monitor.html'] } },
});
