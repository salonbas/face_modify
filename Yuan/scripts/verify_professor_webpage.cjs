const fs=require('fs'),path=require('path'),vm=require('vm');
const {pathToFileURL}=require('url');
const {chromium}=require('C:/Users/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const report=process.argv[2];if(!report)throw new Error('Pass report.html path');
 const html=fs.readFileSync(report,'utf8');
 [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].filter(m=>!m[0].includes('application/json')).forEach(m=>new vm.Script(m[1]));
 const out=path.join(path.dirname(report),'qa');fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--disable-gpu']});
 const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),errors=[],network=[];
 page.on('pageerror',e=>errors.push(e.message));
 await context.route('http://**/*',r=>{network.push(r.request().url());return r.abort();});
 await context.route('https://**/*',r=>{network.push(r.request().url());return r.abort();});
 await page.goto(pathToFileURL(report).href,{waitUntil:'load'});
 await page.evaluate(()=>Promise.all([...document.images].filter(i=>i.src).map(i=>i.decode().catch(()=>{}))));
 const headings=await page.locator('main section h2').allTextContents();
 if(headings.length!==3||!headings[0].includes('A-reference')||!headings[1].includes('Diverse Inputs')||!headings[2].includes('Translation-Invariant'))throw new Error('Topic headings incorrect');
 if(await page.locator('.slide,.nav,#next,#prev,#notesButton,#timer').count())throw new Error('Presentation UI remains');
 if((await page.locator('main').innerText()).includes('Mask'))throw new Error('Unrequested Mask content remains');
 const data=await page.evaluate(()=>JSON.parse(document.getElementById('meeting-data').textContent));
 if(data.methods.length!==4||data.auditedPngCount!==80)throw new Error('Unexpected method scope');
 if(data.methods.map(m=>m.victimSuccess).join(',')!=='8,13,14,16')throw new Error('Result counts incorrect');
 const images=await page.evaluate(()=>[...document.images].filter(i=>i.getAttribute('src')).map(i=>({alt:i.alt,loaded:i.complete&&i.naturalWidth>0})));
 if(images.some(i=>!i.loaded))throw new Error('Image failed to load');
 const desktop=await page.evaluate(()=>({viewport:innerWidth,width:document.documentElement.scrollWidth,visibleSections:[...document.querySelectorAll('main section')].filter(s=>getComputedStyle(s).display!=='none').length}));
 await page.screenshot({path:path.join(out,'web-desktop-top.png')});
 for(const id of ['a-reference','diverse-inputs','translation-invariant'])await page.locator('#'+id).screenshot({path:path.join(out,'web-'+id+'.png')});
 await page.locator('a[href="#diverse-inputs"]').click();
 await page.locator('#demoScale').evaluate(el=>{el.value='90';el.dispatchEvent(new Event('input',{bubbles:true}));});
 if(await page.locator('#scaleValue').innerText()!=='90%')throw new Error('DI illustration control failed');
 await page.locator('#randomPosition').click();
 await page.locator('#demoScale').evaluate(el=>{el.value='100';el.dispatchEvent(new Event('input',{bubbles:true}));});
 const identity=await page.locator('#demoFace').evaluate(el=>({width:el.style.width,left:el.style.left,top:el.style.top}));
 if(identity.width!=='100%'||identity.left!=='0%'||identity.top!=='0%')throw new Error('DI identity illustration failed');
 await page.locator('img[data-zoom-title]').first().click();if(!await page.locator('#imageDialog').isVisible())throw new Error('Image zoom failed');await page.keyboard.press('Escape');
 const positions=[];
 for(const width of [1366,768,390]){
  await page.setViewportSize({width,height:844});
  for(const id of ['a-reference','diverse-inputs','translation-invariant']){
   await page.locator('#'+id).scrollIntoViewIfNeeded();
   positions.push(await page.evaluate(()=>({viewport:innerWidth,width:document.documentElement.scrollWidth})));
  }
 }
 await page.screenshot({path:path.join(out,'web-mobile.png'),fullPage:true});
 if([desktop,...positions].some(s=>s.width>s.viewport+1))errors.push('Horizontal overflow');
 await page.setViewportSize({width:1440,height:1000});await page.emulateMedia({media:'print'});
 if(await page.locator('main section:visible').count()!==3)errors.push('Print hides content');
 const pdf=await page.pdf({format:'A4',printBackground:true,preferCSSPageSize:true});fs.writeFileSync(path.join(out,'web-print-check.pdf'),pdf);
 const printPages=(pdf.toString('latin1').match(/\/Type\s*\/Page\b/g)||[]).length;
 if(network.length)errors.push('Unexpected network requests');
 const result={headings,continuousPage:true,desktop,responsive:positions,images:images.length,networkRequests:network.length,printPages,errors};
 fs.writeFileSync(path.join(out,'webpage-verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 await browser.close();if(errors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
