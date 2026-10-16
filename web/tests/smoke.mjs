#!/usr/bin/env node
/** Browser/GPU integration check. Test-only dependency: Playwright or playwright-core.
 * PLAYWRIGHT_MODULE=/absolute/path/to/playwright-core/index.mjs node web/tests/smoke.mjs
 * CHROME_PATH=/path/to/chrome overrides the installed browser.
 * BASE_URL=https://blackencino.github.io/EncinoWaves/ checks a deployed copy.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, stat, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = resolve(process.env.SMOKE_OUTPUT || resolve(root,
  process.argv.includes('--simulation') ? '../renders/web-simulation' : '../renders/web-smoke'));
const mount = '/EncinoWaves/';
const mime = { '.html':'text/html', '.css':'text/css', '.js':'text/javascript',
  '.mjs':'text/javascript', '.json':'application/json', '.jpg':'image/jpeg',
  '.jpeg':'image/jpeg', '.png':'image/png', '.bin':'application/octet-stream',
  '.woff2':'font/woff2', '.woff':'font/woff', '.svg':'image/svg+xml' };

async function playwright() {
  const flag = process.argv.indexOf('--playwright');
  const candidates = [flag >= 0 ? process.argv[flag+1] : null, process.env.PLAYWRIGHT_MODULE, 'playwright', 'playwright-core'].filter(Boolean);
  for (const candidate of candidates) {
    try { return await import(candidate.startsWith('/') ? pathToFileURL(candidate).href : candidate); }
    catch (error) { if (!['ERR_MODULE_NOT_FOUND','MODULE_NOT_FOUND','ERR_UNSUPPORTED_DIR_IMPORT'].includes(error.code)) throw error; }
  }
  throw new Error('Install playwright-core for tests, or set PLAYWRIGHT_MODULE to its index.mjs. The app itself has no dependency.');
}

async function localServer() {
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      if (!pathname.startsWith(mount)) { response.writeHead(404).end(); return; }
      const file = resolve(root, pathname.slice(mount.length) || 'index.html');
      if (!file.startsWith(root + sep) || !(await stat(file)).isFile()) { response.writeHead(404).end(); return; }
      response.writeHead(200, { 'Content-Type': mime[extname(file)] || 'application/octet-stream', 'Cache-Control':'no-store' });
      response.end(await readFile(file));
    } catch { response.writeHead(404).end(); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  return { server, url: `http://127.0.0.1:${server.address().port}${mount}` };
}

const state = page => page.evaluate(() => window.__school.getState());
const sleep = (page, ms) => page.waitForTimeout(ms);
const problems = [];
const report = { screenshots: [], checks: [], browser: '', url: '', started: new Date().toISOString() };
function check(name) { report.checks.push(name); console.log(`PASS ${name}`); }

function captureProblems(page, label) {
  page.on('pageerror', error => problems.push(`${label}: ${error.message}`));
  page.on('console', message => {
    if (message.type() === 'error') problems.push(`${label} console: ${message.text()}`);
    if (/INVALID_OPERATION|INVALID_ENUM|GL_INVALID|shader.*(?:error|failed)|WebGL.*(?:error|failed)/i.test(message.text()))
      problems.push(`${label} graphics: ${message.text()}`);
  });
  page.on('requestfailed', request => problems.push(`${label} network: ${request.url()} ${request.failure()?.errorText}`));
  page.on('response', response => { if (response.status() >= 400) problems.push(`${label} HTTP ${response.status()}: ${response.url()}`); });
}

async function ready(page, url) {
  await page.goto(url, { waitUntil:'networkidle', timeout:90000 });
  await page.waitForFunction(() => document.body.dataset.ready === 'true' && window.__school?.getState().frames > 2, undefined, { timeout:90000 });
  const status = await state(page);
  assert.equal(status.ready, true);
  assert.ok(!status.error, status.error);
  return status;
}

async function screenshot(page, name) {
  const path = resolve(output, `${name}.png`);
  await page.screenshot({ path, animations:'disabled' });
  report.screenshots.push(path);
}

async function graphicsHealth(page) {
  const result = await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => {
    const canvas = document.querySelector('#ocean');
    const gl = canvas.getContext('webgl2');
    const debug = gl.getExtension('WEBGL_debug_renderer_info');
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let minimum=255, maximum=0, total=0, samples=0;
    const colors = new Set();
    for (let i=0; i<pixels.length; i+=4*101) {
      const luma=(pixels[i]+pixels[i+1]+pixels[i+2])/3;
      minimum=Math.min(minimum,luma); maximum=Math.max(maximum,luma); total+=luma; samples++;
      colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
    }
    resolve({ error:gl.getError(), lost:gl.isContextLost(), width:canvas.width, height:canvas.height,
      minimum, maximum, mean:total/samples, colors:colors.size,
      renderer:debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER) });
  })));
  assert.equal(result.error, 0, `WebGL error ${result.error}`);
  assert.equal(result.lost, false, 'WebGL context lost');
  assert.ok(result.width > 100 && result.height > 100, 'Canvas is empty');
  assert.ok(result.maximum - result.minimum > 15 && result.colors > 20,
    `Renderer produced a blank/uniform frame: ${JSON.stringify(result)}`);
  return result;
}

let server;
let browser;
let activePage;
try {
  await mkdir(output, { recursive:true });
  let url = process.env.BASE_URL;
  if (!url) ({ server, url } = await localServer());
  report.url = url;
  const { chromium } = await playwright();
  const installed = [process.env.CHROME_PATH, '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(p => p && existsSync(p));
  browser = await chromium.launch({ headless:true, ...(installed ? { executablePath:installed } : {}),
    args:['--enable-webgl', '--ignore-gpu-blocklist'] });
  report.browser = browser.version();
  if (process.argv.includes('--simulation')) {
    const page = await browser.newPage();
    captureProblems(page, 'simulation');
    await page.goto(new URL('tests/simulation.html',url).href, { waitUntil:'networkidle' });
    await page.waitForFunction(() => window.testResult, undefined, { timeout:90000 });
    report.simulation = await page.evaluate(() => window.testResult);
    assert.ok(report.simulation.passed, JSON.stringify(report.simulation,null,2));
    assert.deepEqual(problems, [], problems.join('\n'));
    check('Modern and legacy Phillips WebGL FFT, RNG, and phase-continuous edits agree with native Python reference');
    console.log(JSON.stringify(report.simulation,null,2));
    report.passed = true;
  } else if(process.argv.includes('--tessendorf')) {
    const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
    const page=await context.newPage();activePage=page;captureProblems(page,'tessendorf');
    await ready(page,url);
    await page.keyboard.press('t');
    await page.waitForFunction(()=>window.__school.getState().parameters.spectrum==='phillips');
    await sleep(page,250);
    await graphicsHealth(page);
    await screenshot(page,'tessendorf-opening');
    check('T switches the real GPU ocean to Phillips and its simple material');
    const before=(await state(page)).parameters.phillips_amplitude;
    await page.keyboard.press('g');await page.keyboard.down('ArrowRight');await sleep(page,400);await page.keyboard.up('ArrowRight');await sleep(page,500);
    assert.ok((await state(page)).parameters.phillips_amplitude>before);
    await page.keyboard.press('p');await page.keyboard.down('ArrowLeft');await sleep(page,400);await page.keyboard.up('ArrowLeft');await sleep(page,500);
    assert.ok((await state(page)).parameters.phillips_pinch<75);
    check('G gain and P pinch respond to arrow controls');
    await page.locator('#settings-toggle').click();
    assert.equal(await page.locator('#fetch_km-control').isVisible(),false);
    assert.equal(await page.locator('#swell-control').isVisible(),false);
    assert.equal(await page.locator('#foam_amount-control').isVisible(),false);
    assert.equal(await page.locator('#phillips_amplitude-control').isVisible(),true);
    await page.locator('#source-gain').click();
    assert.equal((await state(page)).parameters.phillips_amplitude,55/16);
    await screenshot(page,'tessendorf-source-gain');
    check('Historical controls expose gain and suppress inactive modern-only parameters');
    await page.locator('#scene').selectOption('2');await sleep(page,3200);
    assert.equal((await state(page)).parameters.spectrum,'phillips');
    assert.equal((await state(page)).parameters.phillips_amplitude,55/16);
    await page.locator('#wave-model').selectOption('encino');await sleep(page,250);
    assert.equal((await state(page)).parameters.spectrum,'tma');await graphicsHealth(page);
    assert.equal(await page.locator('#foam_amount-control').isVisible(),true);
    check('Scene changes retain historical mode; Encino mode restores modern controls and rendering');
    await page.locator('#wave-model').selectOption('tessendorf');
    await page.locator('#quality').selectOption('high');
    await page.waitForFunction(()=>window.__school.getState().ready&&window.__school.getState().resolution===1024,undefined,{timeout:90000});
    await graphicsHealth(page);check('Historical mode renders at 1024');
    const phone=await browser.newContext({viewport:{width:390,height:844},deviceScaleFactor:2,isMobile:true,hasTouch:true});
    const mobile=await phone.newPage();activePage=mobile;captureProblems(mobile,'tessendorf-phone');
    const direct=new URL(url);direct.searchParams.set('model','tessendorf');direct.searchParams.set('quality','low');
    await ready(mobile,direct.href);assert.equal((await state(mobile)).parameters.spectrum,'phillips');
    await mobile.locator('#settings-toggle').tap();await screenshot(mobile,'tessendorf-narrow');
    await graphicsHealth(mobile);check('Direct historical link and narrow touch controls render');
    assert.deepEqual(problems,[]);report.passed=true;
  } else {
  const context = await browser.newContext({ viewport:{ width:1440, height:900 }, deviceScaleFactor:1 });
  const page = await context.newPage();
  activePage = page;
  captureProblems(page, 'laptop');
  const initial = await ready(page, url);
  assert.equal(initial.quality, 'medium', 'Laptop default should remain Balanced');
  assert.equal(initial.resolution, 512);
  report.initial = initial;
  report.graphics = await graphicsHealth(page);
  check('Laptop initializes real WebGL 2 rendering without an app error');
  await screenshot(page, 'laptop-opening');
  await sleep(page, 650);
  let current = await state(page);
  assert.ok(current.frames > initial.frames && current.time > initial.time, 'Ocean animation did not advance');
  check('Waves animate');

  await page.locator('#play').click();
  await page.waitForFunction(() => !window.__school.getState().playing);
  const paused = await state(page);
  await sleep(page, 650);
  current = await state(page);
  assert.equal(current.time, paused.time, 'Pause did not freeze ocean time');
  check('Pause freezes wave propagation');
  await page.keyboard.press('Space');
  await page.waitForFunction(() => window.__school.getState().playing);
  check('Space resumes playback');

  const windBefore = (await state(page)).parameters.wind_speed;
  await page.keyboard.press('w');
  await page.keyboard.down('ArrowRight');
  await sleep(page, 700);
  await page.keyboard.up('ArrowRight');
  await sleep(page, 700);
  assert.ok((await state(page)).parameters.wind_speed > windBefore, 'W + right arrow did not increase wind');
  check('School keyboard wind control responds and settles');

  const cameraBefore = (await state(page)).camera;
  await page.keyboard.down('Alt');
  await page.mouse.move(750, 350);
  await page.mouse.down();
  await page.mouse.move(830, 380, { steps:8 });
  await page.mouse.up();
  await page.keyboard.up('Alt');
  assert.notDeepEqual((await state(page)).camera, cameraBefore, 'Alt-drag did not move camera');
  check('Maya Alt-drag camera orbit responds');

  for (let i=1; i<=4; i++) {
    const previous = (await state(page)).scene;
    await page.locator('#next-scene').click();
    await page.waitForFunction(old => window.__school.getState().scene !== old, previous);
    await sleep(page, 3300);
    await graphicsHealth(page);
    await screenshot(page, `laptop-scene-${i}`);
  }
  check('All four School example seas render');

  await page.locator('#settings-toggle').click();
  for (const quality of ['low','medium','high']) {
    await page.locator('#quality').selectOption(quality);
    await page.waitForFunction(value => window.__school.getState().quality === value, quality);
    await page.waitForFunction(() => window.__school.getState().ready && document.body.dataset.ready === 'true', undefined, { timeout:90000 });
    await sleep(page, 200);
    const selected = await state(page);
    assert.equal(selected.qualitySource, 'manual', 'UI quality must be an authoritative manual choice');
    assert.equal(selected.resolution, { low:256, medium:512, high:1024 }[quality]);
    await graphicsHealth(page);
  }
  check('All quality settings resize/rebuild successfully');
  await page.locator('#quality').selectOption('medium');
  await page.waitForFunction(() => window.__school.getState().ready, undefined, { timeout:90000 });
  await page.locator('#wind_speed').evaluate(input => { input.value = 1000*Math.log(22)/Math.log(500); input.dispatchEvent(new Event('input', { bubbles:true })); });
  await page.waitForFunction(() => Math.abs(window.__school.getState().parameters.wind_speed - 22) < .1, undefined, { timeout:10000 });
  check('Settings slider updates the simulated sea');
  await screenshot(page, 'laptop-settings');
  await page.locator('#reset').click();
  await sleep(page, 3300);
  current = await state(page);
  assert.ok(Math.abs(current.parameters.wind_speed - 17) < .2, 'Reset did not restore opening wind');
  check('Reset restores the opening sea');

  const mobileContext = await browser.newContext({ viewport:{ width:390, height:844 }, deviceScaleFactor:2,
    isMobile:true, hasTouch:true, userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1' });
  const mobile = await mobileContext.newPage();
  activePage = mobile;
  captureProblems(mobile, 'narrow');
  const mobileInitial = await ready(mobile, url);
  assert.equal(mobileInitial.quality, 'high', 'An iPhone must start at 1024 quality');
  assert.equal(mobileInitial.resolution, 1024, 'The actual iPhone wave grid must be 1024');
  assert.equal(mobileInitial.qualitySource, 'iphone-auto');
  report.narrow = { initial:mobileInitial, graphics:await graphicsHealth(mobile) };
  check('iPhone identification starts at High quality automatically');
  const layout = await mobile.evaluate(() => ({ viewport:innerWidth, body:document.body.scrollWidth,
    play:document.querySelector('#play').getBoundingClientRect().toJSON() }));
  assert.ok(layout.body <= layout.viewport + 1, `Narrow page overflows horizontally: ${JSON.stringify(layout)}`);
  assert.ok(layout.play.left >= 0 && layout.play.right <= layout.viewport, 'Play control is outside narrow viewport');
  await screenshot(mobile, 'narrow-opening');
  await mobile.locator('#play').tap();
  await mobile.waitForFunction(() => !window.__school.getState().playing);
  await mobile.locator('#settings-toggle').tap();
  await screenshot(mobile, 'narrow-settings');
  check('Narrow touch viewport renders, fits width, and can pause/open settings');

  const lowUrl = new URL(url);
  lowUrl.searchParams.set('quality','low');
  const mobileLow = await ready(mobile, lowUrl.href);
  assert.equal(mobileLow.quality, 'low', 'Explicit low quality must override the iPhone default');
  assert.equal(mobileLow.resolution, 256);
  assert.equal(mobileLow.qualitySource, 'url', 'URL quality must be authoritative');
  report.iphoneExplicitLow = { initial:mobileLow, graphics:await graphicsHealth(mobile) };
  check('An explicit quality=low link overrides the iPhone High default');
  await mobile.locator('#settings-toggle').tap();
  await mobile.locator('#quality').selectOption('high');
  await mobile.waitForFunction(() => window.__school.getState().ready && window.__school.getState().quality === 'high', undefined, { timeout:90000 });
  const manualIphone = await state(mobile);
  assert.equal(manualIphone.resolution, 1024);
  assert.equal(manualIphone.qualitySource, 'manual', 'An iPhone manual choice must disable automatic wave quality changes');
  report.iphoneManualHigh = { state:manualIphone, graphics:await graphicsHealth(mobile) };
  check('An iPhone manual High selection uses 1024 and takes authority over automatic quality');
  await mobileContext.close();

  const widePhoneContext = await browser.newContext({ viewport:{ width:440, height:956 }, deviceScaleFactor:3,
    isMobile:true, hasTouch:true, userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1' });
  const widePhone = await widePhoneContext.newPage();
  activePage = widePhone;
  captureProblems(widePhone, 'wide-iphone');
  const wideInitial = await ready(widePhone, url);
  assert.equal(wideInitial.quality, 'high', 'A wider iPhone viewport must retain the High default');
  assert.equal(wideInitial.resolution, 1024);
  assert.equal(wideInitial.qualitySource, 'iphone-auto');
  report.wideIphone = { initial:wideInitial, graphics:await graphicsHealth(widePhone) };
  await screenshot(widePhone, 'wide-iphone-high');
  check('Wider iPhone identification also starts at High without a model-name allowlist');
  await widePhoneContext.close();

  const androidContext = await browser.newContext({ viewport:{ width:412, height:915 }, deviceScaleFactor:2,
    isMobile:true, hasTouch:true, userAgent:'Mozilla/5.0 (Linux; Android 15; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36' });
  const android = await androidContext.newPage();
  activePage = android;
  captureProblems(android, 'android');
  const androidInitial = await ready(android, url);
  assert.equal(androidInitial.quality, 'low', 'Other small touch devices should retain Light quality');
  assert.equal(androidInitial.resolution, 256);
  assert.equal(androidInitial.qualitySource, 'default');
  report.android = { initial:androidInitial, graphics:await graphicsHealth(android) };
  check('Android touch-device default remains Light quality');
  await androidContext.close();

  const fallback = await context.newPage();
  activePage = fallback;
  await fallback.addInitScript(() => {
    const original=HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext=function(kind,...args) {
      return kind==='webgl2' ? null : original.call(this,kind,...args);
    };
  });
  await fallback.goto(url, { waitUntil:'networkidle' });
  await fallback.locator('#error').waitFor({ state:'visible' });
  assert.match(await fallback.locator('#error-message').innerText(),/WebGL 2/);
  assert.equal((await state(fallback)).ready,false);
  await screenshot(fallback, 'unsupported-webgl');
  check('Unavailable WebGL displays an actionable fallback instead of a blank canvas');

  assert.deepEqual(problems, [], problems.join('\n'));
  check('No uncaught JavaScript errors, failed assets, or WebGL errors');
  report.final = await state(page);
  report.passed = true;
  console.log(`Screenshots and report: ${output}`);
  }
} catch (error) {
  report.passed = false;
  report.error = error.stack || String(error);
  console.error(error);
  await activePage?.screenshot({ path:resolve(output,'failure.png') }).catch(() => {});
  process.exitCode = 1;
} finally {
  report.problems = problems;
  report.finished = new Date().toISOString();
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2) + '\n').catch(() => {});
  await browser?.close();
  if (server) await new Promise(resolve => server.close(resolve));
}
