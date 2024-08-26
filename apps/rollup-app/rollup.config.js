import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import workerLoader from 'rollup-plugin-web-worker-loader';

export default {
  input: 'src/main.ts',
  output: {
    file: 'public/main.js',
    format: 'es'
  },
  plugins: [
    workerLoader(),
    typescript(),
    resolve({ browser: true }),
    commonjs(),
  ]
}; 
