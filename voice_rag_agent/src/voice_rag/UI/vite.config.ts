import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { inspectAttr } from 'kimi-plugin-inspect-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backendTarget = env.VITE_BACKEND_TARGET || 'http://127.0.0.1:8000';

  return {
    base: './',
    plugins: [inspectAttr(), react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      proxy: {
        '/v1': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/healthz': {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
