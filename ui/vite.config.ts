import { defineConfig } from 'vitest/config';
import { modelBridge } from './plugins/model-bridge';

export default defineConfig({
  plugins: [modelBridge()],
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
