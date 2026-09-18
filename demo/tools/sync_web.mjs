#!/usr/bin/env node
/* demo/tools/sync_web.mjs — copy the demo's plain-JS sources into the folder the Next app serves.
 *
 *   node demo/tools/sync_web.mjs            # copy demo/<f> -> web/public/<f>
 *   node demo/tools/sync_web.mjs --check    # report drift and exit 1; changes nothing
 *
 * `demo/` is the source (README, "the plain-JS engine and 3D scene that web/public serves") and
 * `web/public/` is what Vercel ships.  Next serves `public/` verbatim, so the app cannot import out of
 * `demo/` and there is no build step to generate the copy: it is two files, duplicated by hand.  That is
 * the whole reason this script exists — edit `demo/`, run this, commit both.
 *
 * `--check` is the same comparison section 0 of verify_engine_independent.mjs makes, available without a
 * checkpoint or a venv, for a pre-commit hook or CI.
 */
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const FILES = ['engine.js', 'arch3d.js'];
const CHECK = process.argv.includes('--check');

let drift = 0;
for (const f of FILES) {
  const from = path.join(REPO, 'demo', f);
  const to = path.join(REPO, 'web/public', f);
  const src = await readFile(from);
  const dst = await readFile(to).catch(() => null);
  const same = dst !== null && src.equals(dst);

  if (same) {
    console.log(`  ok      demo/${f} == web/public/${f}   ${src.length} B`);
  } else if (CHECK) {
    drift++;
    console.log(`  DRIFT   demo/${f} != web/public/${f}   ${src.length} B vs ${dst === null ? 'missing' : dst.length + ' B'}`);
  } else {
    await writeFile(to, src);
    console.log(`  copied  demo/${f} -> web/public/${f}   ${src.length} B`);
  }
}

if (CHECK && drift) {
  console.log(`\n${drift} file(s) out of sync — run: node demo/tools/sync_web.mjs`);
  process.exit(1);
}
console.log(CHECK ? '\nweb/public is in sync with demo/' : '\ndone — commit both copies');
