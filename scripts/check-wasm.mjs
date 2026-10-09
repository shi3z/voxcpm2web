// Smoke-test a wasm-bindgen bundle before it ships.
//
//   node scripts/check-wasm.mjs web/pkg/voxcpm_rs_bg.wasm
//
// This exists because of a real bug that reached production: an older
// `wasm-opt` reordered the module's two tables and rebound the
// `__wbindgen_externrefs` export to the *funcref* table, which is
// declared with min == max and therefore cannot grow. The generated glue
// does `wasm.__wbindgen_externrefs.grow(4)` during init, so every page
// load failed with:
//
//     WebAssembly.Table.grow(): failed to grow table by 4
//
// The wasm validated fine and the table declarations were byte-identical
// to a working build — only the export binding differed. So the only
// check that catches it is performing the actual operation.
import { readFileSync } from 'node:fs';

const path = process.argv[2];
if (!path) {
  console.error('usage: check-wasm.mjs <file.wasm>');
  process.exit(2);
}

const fail = (msg) => {
  console.error(`FAIL ${path}: ${msg}`);
  process.exit(1);
};

const bytes = readFileSync(path);
let mod;
try {
  mod = await WebAssembly.compile(bytes);
} catch (e) {
  fail(`does not compile: ${e.message}`);
}

// Stub every import so the module can be instantiated outside a browser.
// `__wbindgen_init_externref_table` is itself an import, so stubbing it
// means instantiation does not grow anything — we do that explicitly
// below, which is the point.
const imports = {};
for (const i of WebAssembly.Module.imports(mod)) {
  imports[i.module] ??= {};
  switch (i.kind) {
    case 'function':
      imports[i.module][i.name] = () => {};
      break;
    case 'memory':
      imports[i.module][i.name] = new WebAssembly.Memory({ initial: 1 });
      break;
    case 'table':
      imports[i.module][i.name] = new WebAssembly.Table({ initial: 1, element: 'anyfunc' });
      break;
    case 'global':
      imports[i.module][i.name] = new WebAssembly.Global({ value: 'i32', mutable: true }, 0);
      break;
    default:
      fail(`unhandled import kind ${i.kind} for ${i.module}.${i.name}`);
  }
}

let instance;
try {
  instance = await WebAssembly.instantiate(mod, imports);
} catch (e) {
  fail(`does not instantiate: ${e.message}`);
}

const table = instance.exports.__wbindgen_externrefs;
if (!table) {
  fail('no `__wbindgen_externrefs` export — is this a wasm-bindgen bundle?');
}
if (!(table instanceof WebAssembly.Table)) {
  fail(`\`__wbindgen_externrefs\` is a ${typeof table}, not a Table`);
}

// The exact call the generated glue makes on startup.
const before = table.length;
try {
  table.grow(4);
} catch (e) {
  fail(
    `\`__wbindgen_externrefs\`.grow(4) failed: ${e.message}\n` +
    `       The export is bound to a table of length ${before} that cannot grow — ` +
    `almost certainly the funcref table rather than the externref one.\n` +
    `       This is what a buggy wasm-opt does to a two-table module. ` +
    `Use a newer binaryen, or skip wasm-opt.`
  );
}

// An externref table accepts arbitrary JS values; a funcref table does not.
try {
  table.set(before, {});
} catch (e) {
  fail(`\`__wbindgen_externrefs\` rejects a JS value, so it is not an externref table: ${e.message}`);
}

console.log(
  `ok ${path}: ${(bytes.length / 1048576).toFixed(1)} MB, ` +
  `__wbindgen_externrefs grew ${before} -> ${table.length}`
);
