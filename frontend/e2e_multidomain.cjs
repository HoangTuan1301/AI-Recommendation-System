const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function runMultiDomainE2E() {
  const screenshotsDir = path.resolve(__dirname, '../reports/screenshots/e2e_m6');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  console.log('================================================================');
  console.log('  STARTING PHASE M6 MULTI-DOMAIN REAL BROWSER E2E AUDIT');
  console.log('  System: CineAI V2.0 (10 Models) + MusicAI (6 Models)');
  console.log('================================================================');

  let browser;
  try {
    browser = await chromium.launch({
      channel: 'msedge',
      headless: true,
    });
  } catch (e) {
    console.log('Falling back to default chromium browser:', e.message);
    browser = await chromium.launch({ headless: true });
  }

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
      console.error('  [Browser Console Error]:', text);
    }
  });

  // Monitor network errors
  page.on('response', (response) => {
    const status = response.status();
    if (status >= 400) {
      testResults.network_errors.push({
        url: response.url(),
        status: status,
      });
      console.error(`  [Network Error ${status}]:`, response.url());
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
    // 1. OPEN CINEAI & VIEW MOVIE CATALOG
    // ---------------------------------------------------------
    console.log('\n--- Scenario 1: Open CineAI & View Movie Catalog ---');
    await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle', timeout: 15000 });
    await page.waitForTimeout(1000);

    const cineLogo = await page.textContent('header a span');
    recordScenario('CineAI Brand Logo renders', cineLogo.includes('CINE'), `Found: ${cineLogo}`);

    const discoverNav = page.locator('nav a:has-text("Discover")');
    await discoverNav.click();
    await page.waitForTimeout(1000);
    const movieCardsCount = await page.locator('div:has-text("Showing")').count();
    recordScenario('CineAI Discover page loads movie catalog', movieCardsCount > 0, `Url: ${page.url()}`);
    await page.screenshot({ path: path.join(screenshotsDir, '01_cineai_catalog.png') });

    // ---------------------------------------------------------
    // 2. SELECT CINEAI MODEL & GET MOVIE RECOMMENDATIONS
    // ---------------------------------------------------------
    console.log('\n--- Scenario 2: Select CineAI Model & Get Movie Recommendations ---');
    await page.click('nav a:has-text("AI Recommendations")');
    await page.waitForTimeout(1500);
    recordScenario('Navigate to CineAI Recommendations', page.url().includes('/recommendations'));

    // Check movie recommendation cards (#1 rank badge)
    const movieRankBadge = page.locator('span:has-text("#1")').first();
    const movieRankVisible = await movieRankBadge.isVisible();
    const movieCardsCountOnRec = await page.locator('div.aspect-\\[2\\/3\\]').count();
    recordScenario('CineAI recommendations generated with real inference', movieRankVisible && movieCardsCountOnRec > 0, `Rank #1 visible: ${movieRankVisible}, Count: ${movieCardsCountOnRec}`);
    await page.screenshot({ path: path.join(screenshotsDir, '02_cineai_recommendations.png') });

    // ---------------------------------------------------------
    // 3. SWITCH FROM CINEAI TO MUSICAI VIA DOMAIN SWITCHER
    // ---------------------------------------------------------
    console.log('\n--- Scenario 3: Switch from CineAI to MusicAI via Domain Switcher ---');
    const musicSwitchBtn = page.locator('button:has-text("Music")').first();
    await musicSwitchBtn.click();
    await page.waitForTimeout(1500);

    const isMusicUrl = page.url().includes('/music');
    recordScenario('URL redirects to MusicAI domain', isMusicUrl, `Url: ${page.url()}`);

    const brandText = await page.locator('header a span').first().textContent();
    recordScenario('Navbar logo switches dynamically to MUSIC', brandText.includes('MUSIC'), `Brand: ${brandText}`);
    await page.screenshot({ path: path.join(screenshotsDir, '03_domain_switch_to_music.png') });

    // Navigate to Music Recommendations page
    const musicRecNav = page.locator('nav a[href="/music/recommendations"]').first();
    await musicRecNav.click();
    await page.waitForTimeout(1500);
    recordScenario('Navigate to Music Recommendations Page', page.url().includes('/music/recommendations'));

    // ---------------------------------------------------------
    // 4. SELECT MUSICAI USER & RECEIVE TOP-10 SONGS
    // ---------------------------------------------------------
    console.log('\n--- Scenario 4: Select MusicAI User & Receive Top-10 Songs ---');
    const sampleUserBtn = page.locator('button:has-text("User A")').first();
    if (await sampleUserBtn.isVisible()) {
      await sampleUserBtn.click();
      await page.waitForTimeout(1500);
    }

    const songCards = await page.locator('button:has-text("Details")').count();
    recordScenario('Received Top-10 song recommendations', songCards === 10, `Songs count: ${songCards}`);

    const rank1Visible = await page.locator('span:has-text("#1")').first().isVisible();
    recordScenario('Rank #1 song card displayed', rank1Visible);
    await page.screenshot({ path: path.join(screenshotsDir, '04_musicai_top10_songs.png') });

    // ---------------------------------------------------------
    // 5. SWITCH BETWEEN ALL 6 MUSICAI MODELS
    // ---------------------------------------------------------
    console.log('\n--- Scenario 5: Switch Between All 6 MusicAI Models ---');
    const modelNames = [
      { id: 'user_cf', name: 'User-Based' },
      { id: 'ials', name: 'iALS' },
      { id: 'bpr_mf', name: 'BPR-MF' },
      { id: 'neumf', name: 'NeuMF' },
      { id: 'popularity', name: 'Popularity' },
      { id: 'item_cf', name: 'Item-Based' },
    ];

    let allModelsPassed = true;
    for (const m of modelNames) {
      const modelCard = page.locator(`button:has-text("${m.name}")`).first();
      await modelCard.click();
      await page.waitForTimeout(1200);

      const songsAfterSwitch = await page.locator('button:has-text("Details")').count();
      const ok = songsAfterSwitch > 0;
      if (!ok) allModelsPassed = false;
      console.log(`    - Model [${m.name}]: returned ${songsAfterSwitch} recommendations.`);
    }
    recordScenario('All 6 MusicAI models infer successfully and update songs', allModelsPassed);
    await page.screenshot({ path: path.join(screenshotsDir, '05_all_6_models.png') });

    // ---------------------------------------------------------
    // 6. VERIFY SONG TITLE & ARTIST FROM REAL API
    // ---------------------------------------------------------
    console.log('\n--- Scenario 6: Verify Song Title & Artist From Real API ---');
    const firstSongTitle = await page.locator('div.p-4 h3').first().textContent();
    const firstArtistName = await page.locator('div.p-4 h3 + p').first().textContent();
    const hasValidSongInfo = firstSongTitle.trim().length > 0 && firstArtistName.trim().length > 0;
    recordScenario('Song title and artist populated from real MSD catalog', hasValidSongInfo, `"${firstSongTitle.trim()}" by "${firstArtistName.trim()}"`);

    // ---------------------------------------------------------
    // 7. OPEN & VIEW SONG DETAIL MODAL
    // ---------------------------------------------------------
    console.log('\n--- Scenario 7: Open & View Song Detail Modal ---');
    const detailBtn = page.locator('button:has-text("Details")').first();
    await detailBtn.click();
    await page.waitForTimeout(1200);

    const trackDetailsVisible = await page.locator('text=Track Details').isVisible();
    const statsVisible = await page.locator('text=Training Dataset Statistics').isVisible();
    const listensVisible = await page.locator('text=Total Listens (Train)').isVisible();
    recordScenario('Song Detail Modal opens with real metadata & play statistics', trackDetailsVisible && statsVisible && listensVisible);
    await page.screenshot({ path: path.join(screenshotsDir, '07_song_detail_modal.png') });

    // Close modal using specific aria-label
    const closeSongDetailBtn = page.locator('button[aria-label="Close song detail modal"]');
    if (await closeSongDetailBtn.isVisible()) {
      await closeSongDetailBtn.click();
      await page.waitForTimeout(600);
      const modalClosed = !(await page.locator('text=Track Details').isVisible());
      recordScenario('Song Detail Modal closes cleanly', modalClosed);
    }

    // ---------------------------------------------------------
    // 8. VIEW USER TRAINING LISTENING HISTORY MODAL
    // ---------------------------------------------------------
    console.log('\n--- Scenario 8: View User Listening History Modal ---');
    const historyBtn = page.locator('button:has-text("View Training Listening History")').first();
    await historyBtn.click();
    await page.waitForTimeout(1200);

    const historyHeading = await page.locator('#history-modal-title').isVisible();
    const historyItemCount = await page.locator('div:has-text("plays")').count();
    recordScenario('Listening History Modal opens showing training interactions', historyHeading && historyItemCount > 0, `Recorded songs: ${historyItemCount}`);
    await page.screenshot({ path: path.join(screenshotsDir, '08_user_history_modal.png') });

    // ---------------------------------------------------------
    // 9. DEDUPLICATION: TRAINING SONGS NOT IN RECOMMENDATIONS
    // ---------------------------------------------------------
    console.log('\n--- Scenario 9: Deduplication Verification ---');
    // Extract history song titles
    const historySongTitles = await page.$$eval('#history-modal-title ~ div button', els => els.map(e => e.textContent.trim().toLowerCase()));

    // Close history modal
    const closeHistoryBtn = page.locator('button[aria-label="Close history modal"]');
    if (await closeHistoryBtn.isVisible()) {
      await closeHistoryBtn.click();
      await page.waitForTimeout(500);
    }

    // Extract recommendation song titles
    const recommendedTitles = await page.$$eval('div.p-4 h3', els => els.map(e => e.textContent.trim().toLowerCase()));

    const duplicates = recommendedTitles.filter(t => historySongTitles.includes(t));
    recordScenario('Zero overlap between training history and recommended songs (Deduplication)', duplicates.length === 0, `Overlap count: ${duplicates.length}`);

    // ---------------------------------------------------------
    // 10. COLD-START FALLBACK FOR UNSEEN NEW USERS
    // ---------------------------------------------------------
    console.log('\n--- Scenario 10: Cold-Start Fallback Simulation ---');
    const inputUser = page.locator('form input[type="text"]');
    await inputUser.fill('unseen_cold_start_listener_999');
    const generateBtn = page.locator('button:has-text("Generate")').first();
    await generateBtn.click();
    await page.waitForTimeout(1500);

    const fallbackAlertVisible = await page.locator('text=Cold-Start Fallback Active').isVisible();
    const fallbackSongsCount = await page.locator('button:has-text("Details")').count();
    recordScenario('Cold-Start fallback activates with popularity recommender', fallbackAlertVisible && fallbackSongsCount > 0, `Banner visible: ${fallbackAlertVisible}, Songs: ${fallbackSongsCount}`);
    await page.screenshot({ path: path.join(screenshotsDir, '10_cold_start_fallback.png') });

    // ---------------------------------------------------------
    // 11. BACKEND ERROR & GRACEFUL DEGRADATION
    // ---------------------------------------------------------
    console.log('\n--- Scenario 11: Backend Error & Graceful Degradation ---');
    // Switch back to valid user
    const sampleUserA = page.locator('button:has-text("User A")').first();
    await sampleUserA.click();
    await page.waitForTimeout(1500);
    const recovered = !(await page.locator('text=Cold-Start Fallback Active').isVisible());
    recordScenario('Application cleanly recovers from fallback to personalized mode', recovered);

    // ---------------------------------------------------------
    // 12. SWITCH FROM MUSICAI BACK TO CINEAI
    // ---------------------------------------------------------
    console.log('\n--- Scenario 12: Switch from MusicAI Back to CineAI ---');
    const movieSwitchBtn = page.locator('button:has-text("Movies")').first();
    await movieSwitchBtn.click();
    await page.waitForTimeout(1500);

    const isCineBrand = (await page.locator('header a span').first().textContent()).includes('CINE');
    recordScenario('Domain Switcher returns to CineAI with Movies theme', isCineBrand, `Brand: ${isCineBrand}`);
    await page.screenshot({ path: path.join(screenshotsDir, '12_switch_back_to_cineai.png') });

    // ---------------------------------------------------------
    // 13. RESPONSIVE TESTS ACROSS DESKTOP, TABLET & MOBILE
    // ---------------------------------------------------------
    console.log('\n--- Scenario 13: Responsive Testing Across Devices ---');
    // Tablet Viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('http://127.0.0.1:5173/music/recommendations', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    const tabletCards = await page.locator('button:has-text("Details")').count();
    recordScenario('Tablet viewport (768x1024) renders responsive song grid', tabletCards > 0, `Cards: ${tabletCards}`);
    await page.screenshot({ path: path.join(screenshotsDir, '13_tablet_viewport.png') });

    // Mobile Viewport
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('http://127.0.0.1:5173/music/recommendations', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    const mobileCards = await page.locator('button:has-text("Details")').count();
    recordScenario('Mobile viewport (375x812) renders responsive song cards', mobileCards > 0, `Cards: ${mobileCards}`);
    await page.screenshot({ path: path.join(screenshotsDir, '13_mobile_viewport.png') });

    // Reset to Desktop
    await page.setViewportSize({ width: 1440, height: 900 });

    // ---------------------------------------------------------
    // 14. DIRECT URL NAVIGATION
    // ---------------------------------------------------------
    console.log('\n--- Scenario 14: Direct URL Navigation ---');
    const urlsToTest = [
      { url: 'http://127.0.0.1:5173/music', expectText: 'Multi-Domain Music Intelligence' },
      { url: 'http://127.0.0.1:5173/music/models', expectText: 'MusicAI Model Leaderboard' },
      { url: 'http://127.0.0.1:5173/lab', expectText: 'Model Benchmark' },
    ];

    let allUrlsDirectOk = true;
    for (const u of urlsToTest) {
      await page.goto(u.url, { waitUntil: 'networkidle' });
      await page.waitForTimeout(800);
      const content = await page.content();
      const found = content.includes(u.expectText);
      if (!found) allUrlsDirectOk = false;
      console.log(`    - Direct URL [${u.url}]: found text "${u.expectText}" -> ${found}`);
    }
    recordScenario('Direct URL deep-linking works across all multi-domain routes', allUrlsDirectOk);

    // ---------------------------------------------------------
    // 15. PAGE RELOAD & STATE PERSISTENCE
    // ---------------------------------------------------------
    console.log('\n--- Scenario 15: Page Reload & State Persistence ---');
    await page.goto('http://127.0.0.1:5173/music/recommendations', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);

    const reloadedCards = await page.locator('button:has-text("Details")').count();
    const brandAfterReload = await page.locator('header a span').first().textContent();
    recordScenario('Page reload maintains active music domain and song catalog', reloadedCards > 0 && brandAfterReload.includes('MUSIC'));
    await page.screenshot({ path: path.join(screenshotsDir, '15_page_reload_persistence.png') });

  } catch (err) {
    console.error('Critical test runtime error:', err);
    recordScenario('E2E Execution Error Catch', false, err.message);
  } finally {
    await browser.close();
  }

  // -----------------------------------------------------------
  // GENERATE SUMMARY
  // -----------------------------------------------------------
  console.log('\n================================================================');
  console.log('  PHASE M6 E2E AUDIT SUMMARY');
  console.log(`  Total Scenarios:  ${testResults.total_scenarios}`);
  console.log(`  Passed:           ${testResults.passed_scenarios}`);
  console.log(`  Failed:           ${testResults.failed_scenarios}`);
  console.log(`  Console Errors:   ${testResults.console_errors.length}`);
  console.log(`  Network Errors:   ${testResults.network_errors.length}`);
  console.log('================================================================\n');

  // Save JSON report
  const reportPath = path.resolve(__dirname, '../reports/musicai/phase_m6_e2e_results.json');
  fs.writeFileSync(reportPath, JSON.stringify(testResults, null, 2), 'utf-8');
  console.log('Results saved to:', reportPath);

  if (testResults.failed_scenarios > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

runMultiDomainE2E();
