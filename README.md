**NOTE** Not a Vite problem!

See [solved issue](https://github.com/vitejs/vite/issues/17950) for reference.

In summary, this issue is related to the TS config key `useDefineForClassFields` which is default `true` if `compilerOptions.target` is >= `"es2022"`.

When running [PNext three loader](https://github.com/pnext/three-loader/tree/master) in a Vite bundled app there are issues with the web worker. Specifically the `adaptive` point type is not updated in the render loop, thus creating a `fixed` point type since the shader is not getting proper data.

---

ORIGINAL

# ~~Vite web worker bundling issue~~

## Steps

- `pnpm i` (install)
- `npm run dev` (start dev servers)

### Incorrect

- [localhost:3000](http://localhost:3000) (Vite app)

### Working

- [localhost:8080](http://localhost:8080) (Rollup app)
- [localhost:9000](http://localhost:9000) (Original PNext app with webpack)

To see issue, click `Load` on each page. What you will see is the pointcloud loading with `fixed` point type, when it should be `adaptive`. The way you can distinguish the two is:

- in `fixed` the points will start small and grow
- in `adaptive` the points will start big and shrink

This is simplified but should provide an adequate explanation of what to look for. By running the provided apps and comparing them, side-by-side, the difference will become clear (if their are any doubts).

As far as we can tell, the issue stems from how Vite deals with web workers when doing its "Vite magic".

## Possible Culprit

Looking closer at the final builds (i.e. this issue extends from `dev` to `prod` as well) we have come to the conclusion that their must be something with how Vite is handling the import of the web worker found in `apps/vite-app/src/pnext-loader/loading2/worker-pool.ts`.

In `createWorker` we've tried 2 ways of importing the web worker (both involve changing the `require` reference for proper Vite import handling):

1. `import DecoderWorker from './decoder.worker.js?worker&inline';` (**NOTE** Removing `inline` does not solve the issue. Using it helps us identify the difference in results between the Vite and Rollup bundlers)
2. `return new Worker(new URL('./decoder.worker.js', import.meta.url),{ type: 'module' });`

We've tried to identify when/where Vite deals with the imported worker. Currently we've been looking at `webWorkerPlugin` in the build source code of Vite. Our working theory is that there has to be something in the "wrapping abstractions" in `webWorkerPlugin` that is causing the "final state" to differ in the Vite app from the Rollup/Webpack apps.
