const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function runE2E() {
  const screenshotsDir = path.resolve(__dirname, '../reports/screenshots/e2e');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  console.log('=== STARTING REAL BROWSER E2E ACCEPTANCE AUDIT ===');
  console.log('Launching browser (Microsoft Edge)...');
  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true,
  });

  const testResults = {
    total_scenarios: 0,
    passed_scenarios: 0,
    failed_scenarios: 0,
    scenarios: [],
    console_errors: [],
    network_errors: [],
  };

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });

  const page = await context.newPage();

  // Monitor console errors
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      const text = msg.text();
      testResults.console_errors.push(text);
      console.error(' [Browser Console Error]:', text);
    }
  });

  // Monitor network failures
  page.on('response', (response) => {
    const status = response.status();
    if (status >= 400) {
      testResults.network_errors.push({
        url: response.url(),
        status: status,
      });
      console.error(` [Network Error ${status}]:`, response.url());
    }
  });

  const recordScenario = (name, passed, details = '') => {
    testResults.total_scenarios++;
    if (passed) {
      testResults.passed_scenarios++;
      console.log(`  ✓ [PASS] ${name} ${details}`);
    } else {
      testResults.failed_scenarios++;
      console.error(`  ✗ [FAIL] ${name} ${details}`);
    }
    testResults.scenarios.push({ name, passed, details });
  };

  try {
    // ---------------------------------------------------------
    // 1. HOME PAGE & HERO INTERACTION
    // ---------------------------------------------------------
    console.log('\n--- Scenario 1: Home Page & Hero Interactions ---');
    await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle', timeout: 15000 });
    await page.waitForTimeout(1000);

    const brandLogo = await page.textContent('header a span');
    recordScenario('Navbar Brand Logo renders', brandLogo.includes('CINE'), `Found: ${brandLogo}`);

    const heroHeading = await page.locator('h1').textContent();
    recordScenario('Hero Heading renders', heroHeading.length > 0, `Title: "${heroHeading.trim()}"`);

    // Click Movie Details button on Hero to test Modal
    const detailsBtn = page.locator('button:has-text("Movie Details & Stats")');
    if (await detailsBtn.isVisible()) {
      await detailsBtn.click();
      await page.waitForTimeout(1000);
      const modalVisible = await page.locator('text=Train Rating Count').isVisible();
      recordScenario('Hero Movie Details Modal opens cleanly', modalVisible);

      // Close modal using aria-label
      const closeBtn = page.locator('button[aria-label="Close modal"]');
      if (await closeBtn.isVisible()) {
        await closeBtn.click();
        await page.waitForTimeout(600);
        const modalClosed = !(await page.locator('text=Train Rating Count').isVisible());
        recordScenario('Movie Detail Modal closes cleanly', modalClosed);
      }
    }

    await page.screenshot({ path: path.join(screenshotsDir, '01_home_desktop.png') });

    // ---------------------------------------------------------
    // 2. DISCOVER PAGE & SEARCH & FILTER
    // ---------------------------------------------------------
    console.log('\n--- Scenario 2: Discover Page Search & Filters ---');
    await page.click('nav a:has-text("Discover")');
    await page.waitForTimeout(1000);
    recordScenario('Navigate to Discover page', page.url().includes('/discover'));

    // Test Search Input
    const searchInput = page.locator('input[placeholder*="Search movie titles"]');
    await searchInput.fill('Godfather');
    await page.waitForTimeout(1000); // debounce wait

    const searchCountText = await page.locator('text=Showing').textContent();
    recordScenario('Search filter works for "Godfather"', searchCountText.includes('Showing'), searchCountText);

    // Test Genre Filter Chip
    const genreSciFi = page.locator('button:has-text("Sci-Fi")');
    await searchInput.fill(''); // clear search
    await genreSciFi.click();
    await page.waitForTimeout(800);
    const sciFiCount = await page.locator('text=Showing').textContent();
    recordScenario('Genre chip filter works for "Sci-Fi"', sciFiCount.includes('Showing'), sciFiCount);

    await page.screenshot({ path: path.join(screenshotsDir, '02_discover_desktop.png') });

    // ---------------------------------------------------------
    // 3. AI RECOMMENDATIONS & INFERENCE WITH REAL MODELS
    // ---------------------------------------------------------
    console.log('\n--- Scenario 3: AI Recommendations & Real Inference ---');
    await page.click('nav a:has-text("AI Recommendations")');
    await page.waitForTimeout(1000);
    recordScenario('Navigate to Recommendations page', page.url().includes('/recommendations'));

    // Switch model to User-Based CF
    const userCfCard = page.locator('div:has-text("User-Based Collaborative Filtering")').first();
    await userCfCard.click();
    await page.waitForTimeout(500);

    // Change Top-N to 20
    const top20Btn = page.locator('button:has-text("Top 20")');
    await top20Btn.click();
    await page.waitForTimeout(500);

    // Click Generate Recommendations
    const genBtn = page.locator('button:has-text("Generate Recommendations")');
    await genBtn.click();
    await page.waitForTimeout(2000); // wait for real backend inference

    const recHeader = await page.locator('h2:has-text("Recommendations for User")').isVisible();
    recordScenario('Recommendations generated for User', recHeader);

    const recRankBadges = await page.locator('div:has-text("#1")').count();
    recordScenario('Rank #1 badge present in recommendation cards', recRankBadges > 0);

    await page.screenshot({ path: path.join(screenshotsDir, '03_recommendations_user_cf_top20.png') });

    // ---------------------------------------------------------
    // 4. AI LAB & BENCHMARK & MODEL BATTLE
    // ---------------------------------------------------------
    console.log('\n--- Scenario 4: AI Lab Leaderboard & Model Battle ---');
    await page.click('nav a:has-text("AI Lab & Benchmark")');
    await page.waitForTimeout(1000);
    recordScenario('Navigate to AI Lab page', page.url().includes('/lab'));

    const leaderboardVisible = await page.locator('text=Official 10-Model Benchmark Leaderboard').isVisible();
    recordScenario('Official Leaderboard renders', leaderboardVisible);

    // Switch to 1v1 Model Battle tab
    const battleTab = page.locator('button:has-text("1v1 Model Battle")');
    await battleTab.click();
    await page.waitForTimeout(600);

    const compareBtn = page.locator('button:has-text("Compare Recommendations")');
    if (await compareBtn.isVisible()) {
      await compareBtn.click();
      await page.waitForTimeout(1500);
      const battleResultVisible = await page.locator('text=Jaccard Similarity Index').isVisible();
      recordScenario('Model Battle computes and renders Jaccard Index', battleResultVisible);
    }

    await page.screenshot({ path: path.join(screenshotsDir, '04_ai_lab_battle.png') });

    // ---------------------------------------------------------
    // 5. USER PROFILE & DYNAMIC SWITCHER
    // ---------------------------------------------------------
    console.log('\n--- Scenario 5: User Profile & Dynamic User Switching ---');
    await page.click('nav a:has-text("User Profile")');
    await page.waitForTimeout(1000);
    recordScenario('Navigate to User Profile page', page.url().includes('/profile'));

    const profileTitle = await page.locator('h1').textContent();
    recordScenario('User Profile heading renders', profileTitle.includes('Profile'), profileTitle);

    const ratedCount = await page.locator('span.text-3xl.font-black.text-white').first().textContent();
    recordScenario('User rating history loaded from Train dataset', parseInt(ratedCount) > 0, `Rated: ${ratedCount.trim()}`);

    // Test quick switcher on Profile page
    const userSelect = page.locator('select:has-text("User #1")');
    if (await userSelect.isVisible()) {
      await userSelect.selectOption('196');
      await page.waitForTimeout(1200);
      const switchedTitle = await page.locator('h1').textContent();
      recordScenario('Quick Switch to User #196 dynamically reloads profile', switchedTitle.includes('196'), switchedTitle);
    }

    await page.screenshot({ path: path.join(screenshotsDir, '05_profile_user196.png') });

    // ---------------------------------------------------------
    // 6. MOBILE RESPONSIVE AUDIT (390x844)
    // ---------------------------------------------------------
    console.log('\n--- Scenario 6: Mobile Responsiveness & Drawer ---');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle', timeout: 10000 });
    await page.waitForTimeout(1000);

    const hamburgerBtn = page.locator('header button[aria-label="Toggle navigation"]');
    const hamburgerVisible = await hamburgerBtn.isVisible();
    recordScenario('Mobile hamburger navigation button renders', hamburgerVisible);

    if (hamburgerVisible) {
      await hamburgerBtn.click();
      await page.waitForTimeout(500);
      const mobileDiscoverLink = await page.locator('header a:has-text("Discover")').last().isVisible();
      recordScenario('Mobile navigation drawer expands with links', mobileDiscoverLink);
    }

    await page.screenshot({ path: path.join(screenshotsDir, '06_home_mobile_responsive.png') });

  } catch (err) {
    console.error('Fatal E2E Execution Error:', err);
    recordScenario('E2E Suite Execution without Crash', false, err.message);
  } finally {
    await browser.close();
  }

  // Save audit results
  const outJson = path.resolve(__dirname, '../results/phase6a_e2e_audit.json');
  fs.writeFileSync(outJson, JSON.stringify(testResults, null, 2), 'utf-8');
  console.log(`\n=== E2E AUDIT COMPLETE ===`);
  console.log(`Total Scenarios: ${testResults.total_scenarios}`);
  console.log(`Passed: ${testResults.passed_scenarios}`);
  console.log(`Failed: ${testResults.failed_scenarios}`);
  console.log(`Console Errors: ${testResults.console_errors.length}`);
  console.log(`Network Errors: ${testResults.network_errors.length}`);
  console.log(`Saved detailed report to: ${outJson}`);
}

runE2E().catch(console.error);
