// Leyenda Lab application: tabs, Learn, Practice, Results.
(function(){
  const h = LAB.h, S = LAB.store;
  const state = { tab:'learn', topic: S.get('topic', 'basics'), mode: S.get('mode', 'quiz'), studied: S.get('studied', {}), plan: S.get('plan', {}), best: S.get('best', {}) };
  const topicById = id => LAB.topics.find(t => t.id === id);
  const tabs = ['learn', 'practice', 'results'];

  function setTab(t){ state.tab = t; tabs.forEach(x => { document.getElementById('tab-' + x).hidden = x !== t; document.getElementById('tab-btn-' + x).setAttribute('aria-selected', x === t); }); render(); window.scrollTo({ top:0 }); }
  tabs.forEach(t => document.getElementById('tab-btn-' + t).onclick = () => setTab(t));

  function progress(){ const n = LAB.topics.filter(t => state.studied[t.id]).length; document.getElementById('prog-label').textContent = n + ' / ' + LAB.topics.length + ' studied'; document.getElementById('prog-bar').style.width = (n / LAB.topics.length * 100) + '%';
    document.querySelectorAll('#topic-chips .chip').forEach(c => c.querySelector('.dot').classList.toggle('done', !!state.studied[c.dataset.id])); }

  // ---------- Learn ----------
  function renderLearn(){
    const chips = document.getElementById('topic-chips');
    chips.replaceChildren(...LAB.topics.map(t => h('button', { class:'chip', role:'tab', 'data-id':t.id, 'aria-selected': t.id === state.topic, onclick: () => { state.topic = t.id; S.set('topic', t.id); renderLearn(); } }, h('span', { class:'dot' + (state.studied[t.id] ? ' done' : '') }), t.name)));
    const t = topicById(state.topic) || LAB.topics[0], view = document.getElementById('topic-view');
    const studied = h('label', { class:'chip', style:'cursor:pointer' }, h('input', { type:'checkbox', id:'studied-' + t.id, checked: state.studied[t.id] ? '' : null, onchange: e => { state.studied[t.id] = e.target.checked; S.set('studied', state.studied); progress(); } }), 'Mark as studied');
    const head = h('div', { class:'topic-head' }, h('span', { class:'eyebrow' }, t.eyebrow), h('h2', {}, t.name), h('p', {}, t.intro), h('div', { class:'row' }, studied, h('button', { class:'btn ghost', type:'button', onclick: () => { state.mode = 'quiz'; S.set('mode', 'quiz'); quizTopic = t.id; setTab('practice'); } }, 'Quiz on ' + t.name)));
    const cards = h('div', { class:'cards' }, ...t.cards.map(c => h('div', { class:'card' }, h('h4', {}, c.t), c.d ? h('p', {}, c.d) : null, c.list ? h('ul', {}, ...c.list.map(li => h('li', {}, li))) : null)));
    const jury = h('div', { class:'jury' }, h('b', {}, 'Say it to the jury'), '“' + t.jury + '”');
    const demos = t.demos.map(id => { const d = LAB.demos[id]; if (!d) return null; const ctl = h('div', { class:'controls' }), stage = h('div', { class:'stage' });
      const box = h('div', { class:'demo' }, h('div', { class:'demo-head' }, h('h3', {}, d.title), h('p', {}, d.blurb)), ctl, stage); try { d.render(ctl, stage); } catch (e) { stage.append(h('p', {}, 'This bench could not start: ' + e.message)); } if (!ctl.childElementCount) { ctl.remove(); box.classList.add('one'); } return box; });
    const figs = t.figures.length ? h('div', { class:'figs' }, ...t.figures.map(f => h('figure', { class:'figure' }, h('img', { src:f.src, alt:f.cap.replace(/<[^>]+>/g, ''), loading:'lazy' }), h('figcaption', { html:f.cap })))) : null;
    const idx = LAB.topics.indexOf(t), next = LAB.topics[idx + 1];
    view.replaceChildren(head, h('div', { class:'section-title' }, h('h3', {}, 'Concepts'), h('span', {}, t.cards.length + ' cards')), cards, jury,
      h('div', { class:'section-title' }, h('h3', {}, 'Try it'), h('span', {}, demos.length + ' interactive ' + (demos.length === 1 ? 'bench' : 'benches'))), ...demos,
      figs ? h('div', { class:'section-title' }, h('h3', {}, 'Your figures'), h('span', {}, 'from your notebooks')) : null, figs,
      h('div', { class:'row', style:'margin-top:26px' }, next ? h('button', { class:'btn', type:'button', onclick: () => { state.topic = next.id; S.set('topic', next.id); renderLearn(); window.scrollTo({ top:0 }); } }, 'Next: ' + next.name) : h('button', { class:'btn', type:'button', onclick: () => { state.mode = 'jury'; S.set('mode', 'jury'); setTab('practice'); } }, 'Jury drill')),
      renderPlan());
    progress();
  }
  function renderPlan(){
    const days = [['d1','Day 1 · Basics'],['d2','Day 2 · CNN'],['d3','Day 3 · Autoencoder + Denoising'],['d4','Day 4 · NLP · RNN'],['d5','Day 5 · Transfer + Captioning'],['d6','Day 6 · Full quiz + jury drill'],['d7','Day 7 · Re-read the jury lines, rest']];
    return h('div', { style:'margin-top:30px' }, h('div', { class:'section-title' }, h('h3', {}, 'Your week'), h('span', {}, 'one block a day')), h('div', { class:'plan' }, ...days.map(([k, label]) => h('label', {}, h('input', { type:'checkbox', id:'plan-' + k, checked: state.plan[k] ? '' : null, onchange: e => { state.plan[k] = e.target.checked; S.set('plan', state.plan); } }), label))));
  }

  // ---------- Practice ----------
  let quizTopic = 'all', quizOrder = [], quizPos = 0, quizScore = 0, quizAnswered = false;
  let cardTopic = 'all', cardQueue = [], cardFlip = false, cardGot = 0, cardAgain = 0;
  let juryIdx = 0, juryTimer = null, juryLeft = 60, juryShow = false;
  function shuffle(a){ for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
  const topicSeg = (value, onChange) => LAB.seg([['all', 'All topics'], ...LAB.topics.map(t => [t.id, t.name])], value, onChange);

  function renderPractice(){
    const chips = document.getElementById('mode-chips'), modes = [['quiz','Quiz'],['cards','Flashcards'],['jury','Jury drill']];
    chips.replaceChildren(...modes.map(([id, label]) => h('button', { class:'chip', role:'tab', 'aria-selected': id === state.mode, onclick: () => { state.mode = id; S.set('mode', id); renderPractice(); } }, label)));
    const view = document.getElementById('practice-view'); view.replaceChildren();
    if (state.mode === 'quiz') renderQuiz(view); else if (state.mode === 'cards') renderCards(view); else renderJury(view);
  }
  function startQuiz(){ quizOrder = shuffle(LAB.quiz.map((q, i) => i).filter(i => quizTopic === 'all' || LAB.quiz[i].t === quizTopic)); quizPos = 0; quizScore = 0; quizAnswered = false; }
  function renderQuiz(view){
    if (!quizOrder.length || quizOrder.some(i => quizTopic !== 'all' && LAB.quiz[i].t !== quizTopic)) startQuiz();
    const box = h('div', { class:'quiz' }), seg = topicSeg(quizTopic, v => { quizTopic = v; startQuiz(); renderPractice(); });
    const best = state.best[quizTopic];
    box.append(h('div', { class:'qbar' }, seg, h('span', { class:'pill' }, 'best so far ', h('b', {}, best ? best.score + ' / ' + best.total : '–'))));
    if (quizPos >= quizOrder.length) {
      const total = quizOrder.length, pct = Math.round(quizScore / total * 100);
      box.append(h('div', { class:'qcard' }, h('span', { class:'eyebrow' }, 'round finished'), h('div', { class:'q' }, `${quizScore} / ${total} correct, ${pct}%`), h('p', {}, pct >= 85 ? 'Defence-ready on this set. Switch to the jury drill and say the answers out loud.' : pct >= 60 ? 'Good base. Re-read the cards of the questions you missed, then run the round again.' : 'Go back to Learn for this topic, then try again. Wrong answers are the fastest teacher.'), h('div', { class:'row', style:'margin-top:14px' }, h('button', { class:'btn', type:'button', onclick: () => { startQuiz(); renderPractice(); } }, 'Run again'))));
      view.append(box); return;
    }
    const q = LAB.quiz[quizOrder[quizPos]], card = h('div', { class:'qcard' });
    card.append(h('span', { class:'eyebrow' }, `${topicById(q.t).name} · question ${quizPos + 1} of ${quizOrder.length} · score ${quizScore}`), h('div', { class:'q' }, q.q));
    const opts = h('div', { class:'opts' }), why = h('div', { class:'why' }), nextBtn = h('button', { class:'btn', type:'button', onclick: () => { quizPos++; quizAnswered = false; renderPractice(); } }, quizPos + 1 >= quizOrder.length ? 'See the result' : 'Next question');
    why.hidden = true; nextBtn.hidden = true;
    q.o.forEach((o, k) => opts.append(h('button', { class:'opt', type:'button', onclick: e => { if (quizAnswered) return; quizAnswered = true; const right = k === q.a; if (right) quizScore++;
      opts.querySelectorAll('.opt').forEach((b, j) => { b.disabled = true; if (j === q.a) b.classList.add('right'); else if (j === k) b.classList.add('wrong'); });
      why.textContent = (right ? 'Correct. ' : 'Not this one. ') + q.why; why.hidden = false; nextBtn.hidden = false;
      if (quizPos + 1 >= quizOrder.length) { const b = state.best[quizTopic]; if (!b || quizScore > b.score) { state.best[quizTopic] = { score:quizScore, total:quizOrder.length }; S.set('best', state.best); } } } }, h('span', { class:'k' }, 'ABCD'[k]), o)));
    card.append(opts, why, h('div', { class:'row', style:'margin-top:12px' }, nextBtn));
    box.append(card); view.append(box);
  }
  function startCards(){ cardQueue = shuffle(LAB.cards.filter(c => cardTopic === 'all' || c.t === cardTopic)); cardFlip = false; cardGot = 0; cardAgain = 0; }
  function renderCards(view){
    if (!cardQueue.length && cardGot + cardAgain === 0) startCards();
    const box = h('div', { class:'quiz' }, h('div', { class:'qbar' }, topicSeg(cardTopic, v => { cardTopic = v; startCards(); renderPractice(); }), h('span', { class:'pill' }, 'left ', h('b', {}, cardQueue.length + ''), ' · got it ', h('b', {}, cardGot + ''), ' · again ', h('b', {}, cardAgain + ''))));
    if (!cardQueue.length) { box.append(h('div', { class:'qcard' }, h('span', { class:'eyebrow' }, 'round finished'), h('div', { class:'q' }, `${cardGot} known, ${cardAgain} to repeat`), h('div', { class:'row' }, h('button', { class:'btn', type:'button', onclick: () => { startCards(); renderPractice(); } }, 'New round')))); view.append(box); return; }
    const c = cardQueue[0];
    const flash = h('div', { class:'flash', tabindex:'0', role:'button', 'aria-pressed': cardFlip, onclick: () => { cardFlip = !cardFlip; renderPractice(); }, onkeydown: e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); cardFlip = !cardFlip; renderPractice(); } } },
      h('span', { class:'side' }, (cardFlip ? 'answer' : 'question') + ' · ' + topicById(c.t).name), h('div', { class:'txt' + (cardFlip ? ' ans' : '') }, cardFlip ? c.a : c.q), h('span', { class:'side' }, cardFlip ? 'click to see the question again' : 'click to flip'));
    const again = h('button', { class:'btn ghost', type:'button', onclick: () => { cardQueue.push(cardQueue.shift()); cardAgain++; cardFlip = false; renderPractice(); } }, 'Again'), got = h('button', { class:'btn', type:'button', onclick: () => { cardQueue.shift(); cardGot++; cardFlip = false; renderPractice(); } }, 'Got it');
    box.append(flash, h('div', { class:'row' }, again, got)); view.append(box);
  }
  function renderJury(view){
    const q = LAB.jury[juryIdx % LAB.jury.length];
    const timer = h('span', { class:'timer', id:'jury-timer' }, juryLeft + 's');
    const start = h('button', { class:'btn', type:'button', onclick: () => { clearInterval(juryTimer); juryLeft = 60; timer.textContent = '60s'; juryTimer = setInterval(() => { juryLeft--; timer.textContent = juryLeft + 's'; if (juryLeft <= 0) { clearInterval(juryTimer); timer.textContent = 'time'; } }, 1000); } }, 'Start 60 s and answer aloud');
    const show = h('button', { class:'btn ghost', type:'button', onclick: () => { juryShow = !juryShow; renderPractice(); } }, juryShow ? 'Hide the model answer' : 'Show the model answer');
    const next = h('button', { class:'btn ghost', type:'button', onclick: () => { juryIdx = (juryIdx + 1 + Math.floor(Math.random() * (LAB.jury.length - 1))) % LAB.jury.length; juryShow = false; clearInterval(juryTimer); juryLeft = 60; renderPractice(); } }, 'Another question');
    const box = h('div', { class:'quiz' }, h('div', { class:'qcard' }, h('span', { class:'eyebrow' }, 'the jury asks'), h('div', { class:'q' }, q.q), h('div', { class:'row' }, timer, start, show, next), juryShow ? h('div', { class:'why' }, q.a) : null), h('p', { style:'color:var(--muted);font-size:.9rem' }, 'Answer in one breath: the claim, the number from your notebook, the reason. Then compare with the model answer.'));
    view.append(box);
  }

  // ---------- Results ----------
  const hist = { step1:{ train_loss:[0.4302,0.3184,0.2857,0.2661,0.2524,0.2363,0.2296], test_loss:[0.4571,0.268,0.3149,0.2493,0.2213,0.2785,0.2522], train_acc:[0.8006,0.8653,0.8809,0.893,0.8962,0.9062,0.9079], test_acc:[0.7952,0.889,0.8632,0.9022,0.9147,0.8795,0.8902] },
    step2:{ train_loss:[0.3161,0.2601,0.2454,0.2319,0.2198,0.2123,0.2074,0.2013], test_loss:[0.259,0.2392,0.235,0.2241,0.223,0.2327,0.2227,0.2239], train_acc:[0.9128,0.9296,0.9338,0.9379,0.9411,0.9426,0.944,0.9461], test_acc:[0.9389,0.9397,0.9424,0.9425,0.9511,0.9523,0.9502,0.9536] } };
  function renderResults(){
    const C = LAB.colors(), view = document.getElementById('results-view'), kpi = (v, l, cls) => h('div', { class:'kpi ' + (cls||'') }, h('span', { class:'v' }, v), h('span', { class:'l' }, l));
    const sec = (title, sub) => h('div', { class:'section-title' }, h('h3', {}, title), h('span', {}, sub));
    const l1 = h('div', { class:'kpis' }, kpi('0.9743', 'accuracy, test 20%', 'accent'), kpi('0.9297', 'precision'), kpi('0.9665', 'recall', 'good'), kpi('0.9478', 'F1'), kpi('546,785', 'trainable parameters'), kpi('44 + 20', 'epochs, step 1 + step 2'));
    const l1cpu = h('div', { class:'kpis' }, kpi('0.9502', 'accuracy, 128 px CPU run', 'accent'), kpi('0.9000', 'F1'), kpi('627 min', 'training time'), kpi('8 + 8', 'max epochs per step, patience 2'));
    const c1 = h('div'), c2 = h('div'), c3 = h('div'), c4 = h('div');
    const ser = (st, key, name, color, dash) => ({ name, data: hist[st][key].map((v, i) => [i + 1, v]), color, dash, points:true });
    view.replaceChildren(
      sec('Livrable 1, team notebook on GPU', 'the official numbers'), l1,
      h('p', { style:'max-width:80ch;margin-top:10px' }, 'Step 1, photos against paintings, reached its best test loss 0.0907 at epoch 38 with train accuracy 0.9842 against test 0.9680, a gap of 1.6 points. Step 2 added the three easy categories at a tenth of the learning rate: best test loss 0.1265 at epoch 14, train 0.9857 against test 0.9743. Recall sits above precision because the loss weight of 3.14 favours keeping photos.'),
      sec('Same model reproduced on this laptop', '128 pixels, CPU, 15 October'), l1cpu,
      h('div', { class:'demo', style:'margin-top:12px' }, h('div', { class:'demo-head' }, h('h3', {}, 'Step 1: photo against painting'), h('p', {}, 'Loss and accuracy per epoch, 15,994 training images. Early stopping on the test loss with patience 2 kept epoch 5.')), h('div', { class:'stage' }, c1), h('div', { class:'stage' }, c2)),
      h('div', { class:'demo', style:'margin-top:12px' }, h('div', { class:'demo-head' }, h('h3', {}, 'Step 2: all five categories'), h('p', {}, 'Started from the step 1 weights at learning rate 0.0001. Test accuracy begins at 0.939 instead of 0.80: the curriculum works on CPU too.')), h('div', { class:'stage' }, c3), h('div', { class:'stage' }, c4)),
      sec('Livrable 2, denoising the 148 TouNum images', 'executed against the team classifier'),
      h('div', { class:'kpis' }, kpi('25.57 dB', 'PSNR of the noisy test photos'), kpi('−2.93 dB', 'Model A, plain autoencoder', 'bad'), kpi('+4.18 dB', 'Model B, residual + skips', 'good'), kpi('0.669 → 0.731', 'mean probability of photo after cleaning'), kpi('102 → 115', 'images kept of 148', 'accent'), kpi('0.35 vs 0.90', 'F1, latent features vs supervised')),
      h('div', { class:'demo', style:'margin-top:12px' }, h('div', { class:'demo-head' }, h('h3', {}, 'PSNR on held-out photos'), h('p', {}, 'Anything below the dashed line makes the images worse than they were.')), h('div', { class:'stage' }, (() => { const d = h('div'); setTimeout(() => LAB.barChart(d, { h:230, yMin:0, yMax:34, fmt: v => v.toFixed(1), bars:[{ label:'noisy input', value:25.57, color:C.muted }, { label:'Model A', value:22.65, color:C.bad }, { label:'Model B', value:29.75, color:C.good }], baseline:{ y:25.57, label:'do nothing: 25.57 dB' }, yLabel:'PSNR, dB', aria:'PSNR bars' })); return d; })()), h('div', { class:'stage' }, (() => { const d = h('div'); setTimeout(() => LAB.barChart(d, { h:230, yMin:0, yMax:148, fmt: v => v.toFixed(0), bars:[{ label:'before cleaning', value:102, color:C.muted }, { label:'after Model B', value:115, color:C.accent }], baseline:{ y:148, label:'all 148 images' }, yLabel:'kept as photographs', aria:'kept images' })); return d; })())),
      sec('Workshops', 'the numbers you will quote'),
      h('div', { class:'kpis' }, kpi('99.2% / 63.8%', 'flowers baseline, train / validation', 'bad'), kpi('epoch 4', 'where the flower validation loss turned up'), kpi('0.82 → 0.657', 'GRU accuracy at learning rate 0.01', 'bad'), kpi('≈ 0.80', 'tweets accuracy at 0.001', 'good'), kpi('2,048', 'InceptionV3 features per image'), kpi('64 × 2,048', 'regions × features for captioning')),
      h('div', { class:'figs' }, h('figure', { class:'figure' }, h('img', { src:'img/flowers_overfit.jpg', alt:'flower CNN over-fitting curves', loading:'lazy' }), h('figcaption', { html:'<b>Flowers, no regularisation.</b> The reference picture of over-fitting for the defence.' })), h('figure', { class:'figure' }, h('img', { src:'img/liv2_tounum.jpg', alt:'TouNum images before and after denoising', loading:'lazy' }), h('figcaption', { html:'<b>TouNum before and after Model B.</b> The deliverable in one image.' }))));
    LAB.lineChart(c1, { h:220, series:[ser('step1', 'train_loss', 'train loss', C.accent), ser('step1', 'test_loss', 'test loss', C.second)], xLabel:'epoch', marks:[{ x:5, label:'best, epoch 5' }], aria:'step 1 loss' });
    LAB.lineChart(c2, { h:220, series:[ser('step1', 'train_acc', 'train accuracy', C.accent), ser('step1', 'test_acc', 'test accuracy', C.second)], xLabel:'epoch', fmtY: v => Math.round(v * 100) + '%', aria:'step 1 accuracy' });
    LAB.lineChart(c3, { h:220, series:[ser('step2', 'train_loss', 'train loss', C.accent), ser('step2', 'test_loss', 'test loss', C.second)], xLabel:'epoch', marks:[{ x:7, label:'best, epoch 7' }], aria:'step 2 loss' });
    LAB.lineChart(c4, { h:220, series:[ser('step2', 'train_acc', 'train accuracy', C.accent), ser('step2', 'test_acc', 'test accuracy', C.second)], xLabel:'epoch', fmtY: v => Math.round(v * 100) + '%', aria:'step 2 accuracy' });
  }

  function render(){ if (state.tab === 'learn') renderLearn(); else if (state.tab === 'practice') renderPractice(); else renderResults(); }

  // Theme changes redraw the charts with the new tokens.
  try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render); new MutationObserver(render).observe(document.documentElement, { attributes:true, attributeFilter:['data-theme'] }); } catch (e) {}
  const hash = (location.hash || '').slice(1);
  if (tabs.includes(hash)) state.tab = hash; else if (topicById(hash)) { state.topic = hash; }
  setTab(state.tab);
})();
