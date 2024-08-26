import { defineConfig } from "vite"
import { viteCommonjs } from '@originjs/vite-plugin-commonjs';
import rawLoader from 'vite-raw-plugin';

export default defineConfig({
    plugins: [
        viteCommonjs(),
        rawLoader({
            fileRegex: /\.(vert|frag)$/
        }),
    ],
    server: {
        port: 3000,
        host: "localhost",
    },
})