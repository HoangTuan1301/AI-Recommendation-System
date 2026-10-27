const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function run() {
  const outputDir = path.resolve(__dirname, '../reports/screenshots');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log('Launching browser (msedge)...');
  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true,
  });

  const consoleLogs = [];

  const pagesToCapture = [
    { name: 'home-desktop', url: 'http://localhost:5173/', viewport: { width: 1440, height: 900 } },
    { name: 'home-mobile', url: 'http://localhost:5173/', viewport: { width: 390, height: 844 } },
    { name: 'discover-desktop', url: 'http://localhost:5173/discover', viewport: { width: 1440, height: 900 } },
    { name: 'recommendations-desktop', url: 'http://localhost:5173/recommendations', viewport: { width: 1440, height: 900 } },
    { name: 'ai-lab-desktop', url: 'http://localhost:5173/lab', viewport: { width: 1440, height: 900 } },
    { name: 'profile-desktop', url: 'http://localhost:5173/profile', viewport: { width: 1440, height: 900 } },
  ];

  for (const item of pagesToCapture) {
    console.log(`Capturing ${item.name} (${item.url})...`);
    const page = await browser.newPage({ viewport: item.viewport });
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleLogs.push(`[${item.name}] ERROR: ${msg.text()}`);
      }
    });

    try {
      await page.goto(item.url, { waitUntil: 'networkidle', timeout: 15000 });
      await page.waitForTimeout(1000); // Allow smooth transitions / images to render
      const screenshotPath = path.join(outputDir, `${item.name}.png`);
      await page.screenshot({ path: screenshotPath, fullPage: false });
      console.log(`Saved screenshot: ${screenshotPath}`);
    } catch (err) {
      console.error(`Failed to capture ${item.name}:`, err.message);
    } finally {
      await page.close();
    }
  }

  await browser.close();
  console.log('Visual QA capture complete.');
  if (consoleLogs.length > 0) {
    console.log('Console Errors detected:\n', consoleLogs.join('\n'));
  } else {
    console.log('Zero console errors detected!');
  }
}

run().catch(console.error);
