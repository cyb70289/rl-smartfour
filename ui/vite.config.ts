import type { Connect } from 'vite';
import { defineConfig, type Plugin } from 'vitest/config';
import { modelBridge } from './plugins/model-bridge';

// SMARTFOUR_VIEW=openbook starts the UI in read-only opening-book viewer
// mode: play controls are hidden and the panel lists the book states.
const openbookView = process.env.SMARTFOUR_VIEW === 'openbook';

/** Serves the view mode decided at server-start time; the UI boots into
 * game or opening-book viewer accordingly. Works on dev and preview
 * without rebuilding (an HTML-injected flag would be baked by vite build). */
function viewModeEndpoint(): Plugin {
  const handler: Connect.NextHandleFunction = (_req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ openbookView }));
  };
  return {
    name: 'view-mode-endpoint',
    configureServer(server) {
      server.middlewares.use('/api/viewmode', handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use('/api/viewmode', handler);
    },
  };
}

export default defineConfig({
  plugins: [modelBridge(), viewModeEndpoint()],
  build: {
    rollupOptions: {
      output: {
        // Split the three.js library into its own chunk: it only changes on
        // a dependency bump, so app rebuilds keep it browser-cached and the
        // board paints without re-downloading half a megabyte.
        manualChunks: { three: ['three'] },
      },
    },
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
