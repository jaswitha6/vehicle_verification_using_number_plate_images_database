const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const srcDir = path.join(repoRoot, 'frontend');
const outDir = path.join(repoRoot, 'dist');
const apiBase = (process.env.API_BASE_URL || '/api').trim() || '/api';

fs.rmSync(outDir, { recursive: true, force: true });
fs.mkdirSync(outDir, { recursive: true });

for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
  const srcPath = path.join(srcDir, entry.name);
  const destPath = path.join(outDir, entry.name);

  if (entry.isDirectory()) {
    fs.cpSync(srcPath, destPath, { recursive: true });
    continue;
  }

  if (entry.name === 'config.js') {
    const configContents = [
      'window.__APP_CONFIG__ = {',
      `  API_BASE: ${JSON.stringify(apiBase)}`,
      '};',
      ''
    ].join('\n');
    fs.writeFileSync(destPath, configContents, 'utf8');
    continue;
  }

  fs.copyFileSync(srcPath, destPath);
}
