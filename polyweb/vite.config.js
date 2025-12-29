import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'js/dynamic-wallet.js'),
      name: 'DynamicWallet',
      fileName: 'dynamic-wallet',
      formats: ['iife']
    },
    outDir: 'js/dist',
    emptyOutDir: false,
    rollupOptions: {
      output: {
        // Expose as global variable
        extend: true
      }
    }
  },
  define: {
    'process.env': {}
  }
});
