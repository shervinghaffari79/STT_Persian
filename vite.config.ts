import path from "path";
import { fileURLToPath } from "url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    // Only the frontend is meant to be network/internet-exposed: bind to all
    // interfaces on :5000 (override with FRONTEND_PORT). The FastAPI backend
    // (backend/server.py) stays on 127.0.0.1 -- never exposed directly -- and
    // is reached only through the proxy below, which runs on this same
    // machine regardless of who is connecting to :5000.
    host: true,
    port: Number(process.env.FRONTEND_PORT) || 5000,
    strictPort: true,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${process.env.BACKEND_PORT || 8000}`,
        changeOrigin: true,
      },
    },
  },
});
