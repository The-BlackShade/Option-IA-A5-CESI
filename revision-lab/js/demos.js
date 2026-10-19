// Interactive benches, part 1: activations, gradient descent, over-fitting, threshold, margin, convolution.
window.LAB = window.LAB || {};
LAB.demos = LAB.demos || {};
(function(){
  const h = LAB.h, pill = (label, value, cls) => h('span', { class:'pill ' + (cls||'') }, label + ' ', h('b', {}, value));

  LAB.demos.activations = { title:"Activation functions", blurb:"Move x and read the output. ReLU is a hinge, sigmoid squashes to 0..1, tanh to −1..1. Softmax takes several numbers and shares 100% between them.",
    render(ctl, stage){
      let fn = 'relu', x = 1.5, z = [2.0, 0.5, -1.0];
      const chart = h('div'), read = h('div', { class:'readout' });
      const seg = LAB.seg([['relu','ReLU'],['sigmoid','Sigmoid'],['tanh','tanh'],['softmax','Softmax']], fn, v => { fn = v; draw(); });
      const rx = LAB.range('x', 'act-x', -6, 6, 0.1, x, v => (+v).toFixed(1)); rx.input.oninput = () => { x = +rx.input.value; draw(); };
      const rz = z.map((v, i) => { const r = LAB.range('logit z' + (i+1), 'act-z' + i, -4, 4, 0.1, v, q => (+q).toFixed(1)); r.input.oninput = () => { z[i] = +r.input.value; draw(); }; return r; });
      ctl.append(seg, rx, ...rz); stage.append(chart, read);
      const f = { relu: v => Math.max(0, v), sigmoid: v => 1 / (1 + Math.exp(-v)), tanh: v => Math.tanh(v) };
      function draw(){
        const C = LAB.colors(); rx.hidden = fn === 'softmax'; rz.forEach(r => r.hidden = fn !== 'softmax');
        if (fn === 'softmax') {
          const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a,b) => a+b, 0), p = e.map(v => v / s);
          LAB.barChart(chart, { h:220, yMin:0, yMax:1, fmt: v => v.toFixed(2), bars: p.map((v,i) => ({ label:'class ' + (i+1) + ' (z=' + z[i].toFixed(1) + ')', value:v })), aria:'softmax probabilities' });
          read.replaceChildren(pill('sum of probabilities', p.reduce((a,b)=>a+b,0).toFixed(2), 'good'), pill('largest logit wins', 'class ' + (z.indexOf(Math.max(...z)) + 1)));
        } else {
          const data = []; for (let v = -6; v <= 6.001; v += 0.1) data.push([v, f[fn](v)]);
          LAB.lineChart(chart, { h:230, series:[{ name:fn, data, color:C.accent }, { name:'your x', data:[[x, f[fn](x)]], color:C.second, points:true }], xLabel:'input x', yLabel:'output', marks:[{ x, label:'x = ' + x.toFixed(1), color:C.second }], aria:fn + ' curve' });
          const slope = fn === 'relu' ? (x > 0 ? 1 : 0) : fn === 'sigmoid' ? f.sigmoid(x) * (1 - f.sigmoid(x)) : 1 - Math.tanh(x) ** 2;
          read.replaceChildren(pill('output', f[fn](x).toFixed(3)), pill('slope here', slope.toFixed(3), slope < 0.05 ? 'warn' : ''), h('span', { class:'pill' }, slope < 0.05 ? 'Flat slope: the gradient that passes back through this neuron is almost zero.' : 'A healthy slope: gradients pass through.'));
        }
      }
      draw();
    } };

  LAB.demos.descent = { title:"Gradient descent and the learning rate", blurb:"The ball starts at w = −3 and takes 30 steps downhill on the loss curve. Change the step size and watch it crawl, converge, bounce, or fly away. Your GRU at 0.01 was the bouncing case.",
    render(ctl, stage){
      let lr = 0.15, shape = 'bowl';
      const chart = h('div'), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const r = LAB.range('learning rate', 'gd-lr', 0.01, 1.2, 0.01, lr, v => (+v).toFixed(2)); r.input.oninput = () => { lr = +r.input.value; draw(); };
      const seg = LAB.seg([['bowl','Smooth bowl'],['bumpy','Valley with bumps']], shape, v => { shape = v; draw(); });
      ctl.append(r, seg); stage.append(chart, read, verdict);
      const L = w => shape === 'bowl' ? (w - 2) ** 2 : (w - 2) ** 2 / 2 + 0.6 * Math.cos(3 * w) + 0.6;
      const dL = w => shape === 'bowl' ? 2 * (w - 2) : (w - 2) - 1.8 * Math.sin(3 * w);
      function draw(){
        const C = LAB.colors(), curve = [], path = []; let w = -3;
        for (let v = -5; v <= 7.001; v += 0.05) curve.push([v, L(v)]);
        for (let s = 0; s <= 30; s++) { path.push([w, L(w)]); w = w - lr * dL(w); if (!isFinite(w) || Math.abs(w) > 1e6) break; }
        const last = path[path.length - 1], inRange = path.filter(p => Math.abs(p[0]) <= 7);
        LAB.lineChart(chart, { h:260, yMin:0, yMax:28, xMin:-5, xMax:7, series:[{ name:'loss L(w)', data:curve, color:C.muted }, { name:'steps', data:inRange, color:C.second, points:true, width:1.5 }], xLabel:'weight w', yLabel:'loss', marks:[{ x:2, label:'minimum', color:C.good }], aria:'gradient descent path' });
        const osc = path.length > 3 && Math.sign(path[path.length-1][0] - 2) !== Math.sign(path[path.length-2][0] - 2);
        let v, cls = '';
        if (!isFinite(last[1]) || last[1] > 1e3) { v = 'Diverged: each step overshoots further than the last. This is the learning-rate-too-high failure, the weights become garbage.'; cls = 'bad'; }
        else if (Math.abs(last[0] - 2) < 0.05) { v = osc ? 'Converged, but by bouncing from side to side: a smaller rate would be calmer.' : 'Converged in 30 steps: the step size suits this curve.'; cls = 'good'; }
        else if (shape === 'bumpy' && Math.abs(dL(last[0])) < 0.3 && Math.abs(last[0] - 2) > 0.4) { v = 'Stuck in a small side valley: the slope there is nearly zero. Momentum, in Adam, is what carries the ball over such bumps.'; cls = 'warn'; }
        else if (osc) { v = 'Bouncing across the valley without settling: too high. Divide the rate by 10.'; cls = 'warn'; }
        else { v = 'Still far from the minimum after 30 steps: too low, training crawls.'; cls = 'warn'; }
        read.replaceChildren(pill('final w', last[0].toFixed(2)), pill('final loss', isFinite(last[1]) ? last[1].toFixed(3) : '∞', cls));
        verdict.textContent = v;
      }
      draw();
    } };

  LAB.demos.overfit = { title:"Over-fitting simulator", blurb:"Three dials shape a training run: how big the model is compared with the data, how much regularisation it gets, and how many images it sees. The dashed line is where early stopping would halt. The two presets reproduce your own runs.",
    render(ctl, stage){
      let cap = 0.9, reg = 0.05, data = 0.3;
      const lossEl = h('div'), accEl = h('div'), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const rc = LAB.range('model size vs data', 'of-cap', 0.1, 1, 0.01, cap, v => (+v).toFixed(2)), rr = LAB.range('regularisation (dropout, augmentation)', 'of-reg', 0, 1, 0.01, reg, v => (+v).toFixed(2)), rd = LAB.range('training data', 'of-data', 0.1, 1, 0.01, data, v => (+v).toFixed(2));
      rc.input.oninput = () => { cap = +rc.input.value; draw(); }; rr.input.oninput = () => { reg = +rr.input.value; draw(); }; rd.input.oninput = () => { data = +rd.input.value; draw(); };
      const presets = h('div', { class:'seg' },
        h('button', { type:'button', onclick: () => set(0.95, 0.02, 0.2) }, 'Preset: flowers baseline'),
        h('button', { type:'button', onclick: () => set(0.35, 0.6, 0.95) }, 'Preset: team classifier'),
        h('button', { type:'button', onclick: () => set(0.12, 0.3, 0.6) }, 'Preset: too small a model'));
      function set(c, r, d){ cap = c; reg = r; data = d; rc.input.value = c; rr.input.value = r; rd.input.value = d; [rc, rr, rd].forEach(x => x.input.dispatchEvent(new Event('input'))); }
      ctl.append(rc, rr, rd, presets); stage.append(lossEl, accEl, read, verdict);
      function draw(){
        const C = LAB.colors(), E = 30, tr = [], va = [], tra = [], vaa = [];
        const fit = cap * (0.5 + 0.5 * data), kT = 0.10 + 0.30 * fit, floorT = 0.03 + 0.22 * reg + 0.45 * (1 - cap);
        const memo = cap * (1 - reg) * (1 - data), gap = 0.06 + 0.9 * memo, rise = 0.0045 * memo, e0 = 4 + 10 * (1 - memo);
        let best = 1e9, bestE = 1;
        for (let e = 1; e <= E; e++) {
          const t = floorT + (1.6 - floorT) * Math.exp(-kT * e);
          const v = floorT + gap + (1.5 - floorT - gap) * Math.exp(-kT * 0.9 * e) + rise * Math.max(0, e - e0) ** 2;
          tr.push([e, t]); va.push([e, v]); tra.push([e, Math.min(0.995, 1 - 0.42 * t)]); vaa.push([e, Math.min(0.995, 1 - 0.42 * v)]);
          if (v < best) { best = v; bestE = e; }
        }
        LAB.lineChart(lossEl, { h:230, yMin:0, yMax:1.7, series:[{ name:'train loss', data:tr, color:C.accent }, { name:'validation loss', data:va, color:C.second }], xLabel:'epoch', marks:[{ x:bestE, label:'early stop, epoch ' + bestE }], aria:'loss curves' });
        LAB.lineChart(accEl, { h:200, yMin:0.3, yMax:1, series:[{ name:'train accuracy', data:tra, color:C.accent }, { name:'validation accuracy', data:vaa, color:C.second }], xLabel:'epoch', fmtY: v => Math.round(v*100) + '%', aria:'accuracy curves' });
        const gapAtEnd = tra[E-1][1] - vaa[E-1][1], rising = va[E-1][1] - best, both = vaa[bestE-1][1];
        let v, cls;
        if (rising > 0.08 || gapAtEnd > 0.12) { v = `Over-fitting (high variance): the gap reaches ${(gapAtEnd*100).toFixed(0)} points and the validation loss turns upward after epoch ${bestE}. Add data, augmentation or dropout, or stop at epoch ${bestE}.`; cls = 'bad'; }
        else if (both < 0.78) { v = 'Under-fitting (high bias): both curves stay poor and close together. The model is too small or trained too little; make it bigger or train longer.'; cls = 'warn'; }
        else { v = `Healthy: a small gap of ${(gapAtEnd*100).toFixed(1)} points and a validation loss that plateaus instead of rising. This is the shape of the team classifier's curves.`; cls = 'good'; }
        read.replaceChildren(pill('best epoch', bestE + ''), pill('gap at the end', (gapAtEnd*100).toFixed(1) + ' pts', cls), pill('validation loss rise after best', rising.toFixed(2), rising > 0.08 ? 'bad' : 'good'));
        verdict.textContent = v;
      }
      draw();
    } };

  // Normal CDF for the threshold bench.
  const erf = x => { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; };
  const cdf = (x, mu, sd) => 0.5 * (1 + erf((x - mu) / (sd * Math.SQRT2)));
  LAB.demos.threshold = { title:"Threshold, precision and recall", blurb:"Two bell curves of classifier scores, 2,000 photos and 6,280 others as in the 20% test set, shaped to reproduce the team model at threshold 0.5. Slide the threshold and watch precision and recall trade places. Switch the class weight off to see what the 1:3 imbalance does on its own.",
    render(ctl, stage){
      let t = 0.5, weighted = true; const nP = 2000, nO = 6280, sd = 0.14;
      const chart = h('div'), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const r = LAB.range('decision threshold', 'th-t', 0.05, 0.95, 0.01, t, v => (+v).toFixed(2)); r.input.oninput = () => { t = +r.input.value; draw(); };
      const seg = LAB.seg([['on','pos_weight 3.14 on'],['off','no class weight']], 'on', v => { weighted = v === 'on'; draw(); });
      ctl.append(r, seg); stage.append(chart, read, verdict);
      function draw(){
        const C = LAB.colors(), muP = weighted ? 0.76 : 0.66, muO = weighted ? 0.22 : 0.17;
        const TP = nP * (1 - cdf(t, muP, sd)), FN = nP - TP, FP = nO * (1 - cdf(t, muO, sd)), TN = nO - FP;
        const prec = TP / (TP + FP || 1), rec = TP / nP, f1 = 2 * prec * rec / (prec + rec || 1), acc = (TP + TN) / (nP + nO);
        const dens = (mu, n) => { const d = []; for (let x = 0; x <= 1.001; x += 0.01) d.push([x, n * Math.exp(-((x - mu) ** 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI)) / 100]); return d; };
        LAB.lineChart(chart, { h:230, yMin:0, series:[{ name:'others (6,280)', data:dens(muO, nO), color:C.muted, area:true }, { name:'photos (2,000)', data:dens(muP, nP), color:C.accent, area:true }], xLabel:'score given by the model', yLabel:'images per 0.01 of score', marks:[{ x:t, label:'threshold ' + t.toFixed(2), color:C.second }], legend:true, aria:'score distributions' });
        read.replaceChildren(pill('precision', prec.toFixed(3), prec < 0.85 ? 'warn' : 'good'), pill('recall', rec.toFixed(3), rec < 0.9 ? 'warn' : 'good'), pill('F1', f1.toFixed(3)), pill('accuracy', acc.toFixed(3)), pill('photos lost', Math.round(FN) + '', FN > 150 ? 'bad' : ''), pill('paintings let through', Math.round(FP) + '', FP > 400 ? 'bad' : ''));
        verdict.textContent = t > 0.6 ? `A high threshold buys precision (${prec.toFixed(2)}) by losing ${Math.round(FN)} real photos. For TouNum each lost photo is never captioned, so this is the wrong direction.` :
          t < 0.4 ? `A low threshold catches almost every photo (recall ${rec.toFixed(2)}) but lets ${Math.round(FP)} non-photos through to the denoiser. Cheap, but it adds up.` :
          weighted ? 'Around 0.5 with the class weight: recall 0.97 above precision 0.93, the team model’s operating point. Always answering not-photo would score 76% accuracy here, which is why accuracy alone is not reported.' :
          'Without the class weight the photo scores sit lower: recall drops and precision rises. The 1:3 imbalance makes the model timid about the rare class.';
      }
      draw();
    } };

  LAB.demos.margin = { title:"How many test images do you need?", blurb:"The score measured on n images is only known to about ±1.96·√(p(1−p)/n). Slide n and p. The three marks are the 148 TouNum images, a round thousand, and the 8,280 images of the Livrable 1 test set.",
    render(ctl, stage){
      let n = 1000, p = 0.9;
      const chart = h('div'), read = h('div', { class:'readout' });
      const rn = LAB.range('test images n', 'mg-n', 1.7, 4.3, 0.01, Math.log10(n), v => Math.round(10 ** v).toLocaleString()), rp = LAB.range('measured accuracy p', 'mg-p', 0.5, 0.99, 0.01, p, v => (+v).toFixed(2));
      rn.input.oninput = () => { n = Math.round(10 ** rn.input.value); draw(); }; rp.input.oninput = () => { p = +rp.input.value; draw(); };
      ctl.append(rn, rp); stage.append(chart, read);
      const marg = (nn, pp) => 1.96 * Math.sqrt(pp * (1 - pp) / nn) * 100;
      function draw(){
        const C = LAB.colors(), d = []; for (let e = 1.7; e <= 4.3; e += 0.02) d.push([e, marg(10 ** e, p)]);
        LAB.lineChart(chart, { h:230, yMin:0, series:[{ name:'margin at p = ' + p.toFixed(2), data:d, color:C.accent }, { name:'your n', data:[[Math.log10(n), marg(n, p)]], color:C.second, points:true }], xLabel:'test images (log scale)', yLabel:'± points', fmtX: v => Math.round(10 ** v).toLocaleString(), xTicks:5, marks:[{ x:Math.log10(148), label:'148' }, { x:3, label:'1,000' }, { x:Math.log10(8280), label:'8,280' }], aria:'margin of error' });
        read.replaceChildren(pill('margin', '± ' + marg(n, p).toFixed(1) + ' points', marg(n, p) > 3 ? 'warn' : 'good'), pill('148 images', '± ' + marg(148, p).toFixed(1)), pill('8,280 images', '± ' + marg(8280, p).toFixed(1)));
      }
      draw();
    } };

  LAB.demos.conv = { title:"Convolution, ReLU and max pooling", blurb:"Click the pixels of the small image to paint (0, ½, 1). The 3×3 filter slides over it and writes one number per position. Pick an edge filter and watch which lines light up, then see what 2×2 max pooling keeps.",
    render(ctl, stage){
      const N = 7; let img = [[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[1,1,1,1,1,1,1],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0]];
      const filters = { vx:[[-1,0,1],[-2,0,2],[-1,0,1]], hy:[[-1,-2,-1],[0,0,0],[1,2,1]], blur:[[1,1,1],[1,1,1],[1,1,1]].map(r => r.map(v => v/9)), sharp:[[0,-1,0],[-1,5,-1],[0,-1,0]] };
      let fk = 'vx', pad = 'valid', relu = true;
      const seg = LAB.seg([['vx','Vertical edges'],['hy','Horizontal edges'],['blur','Blur'],['sharp','Sharpen']], fk, v => { fk = v; draw(); });
      const segP = LAB.seg([['valid','padding valid (shrinks)'],['same','padding same']], pad, v => { pad = v; draw(); });
      const segR = LAB.seg([['on','ReLU on'],['off','ReLU off']], 'on', v => { relu = v === 'on'; draw(); });
      const presets = h('div', { class:'seg' }, h('button', { type:'button', onclick: () => { img = img.map((r, y) => r.map((v, x) => x === 3 || y === 3 ? 1 : 0)); draw(); } }, 'Cross'), h('button', { type:'button', onclick: () => { img = img.map((r, y) => r.map((v, x) => x >= 3 ? 1 : 0)); draw(); } }, 'Half bright'), h('button', { type:'button', onclick: () => { img = img.map((r, y) => r.map((v, x) => x === y ? 1 : 0)); draw(); } }, 'Diagonal'));
      const filtEl = h('div'), imgEl = h('div'), outEl = h('div'), poolEl = h('div'), read = h('div', { class:'readout' });
      ctl.append(seg, segP, segR, h('div', { class:'ctl' }, h('label', {}, 'example images'), presets), h('div', { class:'ctl' }, h('label', {}, 'the filter (weights)'), filtEl));
      stage.append(h('div', { class:'four' }, fig('input image, click to paint', imgEl), fig('feature map', outEl), fig('after 2×2 max pooling', poolEl)), read);
      function fig(cap, el){ return h('figure', {}, el, h('figcaption', {}, cap)); }
      function cellColor(v, maxAbs){ const C = LAB.colors(); const a = Math.min(1, Math.abs(v) / (maxAbs || 1)); return `color-mix(in srgb, ${v >= 0 ? C.accent : C.bad} ${Math.round(a * 85)}%, ${C.surface})`; }
      function grid(el, M, opts){ const maxAbs = Math.max(...M.flat().map(Math.abs), 1e-9); el.className = 'grid-cells'; el.style.gridTemplateColumns = `repeat(${M[0].length}, 30px)`; el.replaceChildren(...M.flatMap((row, y) => row.map((v, x) => { const c = h('div', { class:'cell' + (opts.click ? ' click' : ''), title:v.toFixed(2) }, Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)); c.style.background = cellColor(v, opts.fixed ? 1 : maxAbs); if (opts.click) c.onclick = () => { img[y][x] = img[y][x] === 0 ? 0.5 : img[y][x] === 0.5 ? 1 : 0; draw(); }; return c; }))); }
      function draw(){
        const F = filters[fk], P = pad === 'same' ? 1 : 0, S = N + 2 * P - 2, out = [];
        for (let y = 0; y < S; y++) { out.push([]); for (let x = 0; x < S; x++) { let s = 0; for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) { const yy = y + j - P, xx = x + i - P; const v = yy < 0 || xx < 0 || yy >= N || xx >= N ? 0 : img[yy][xx]; s += v * F[j][i]; } out[y].push(relu ? Math.max(0, s) : s); } }
        const PS = Math.floor(S / 2), pool = []; for (let y = 0; y < PS; y++) { pool.push([]); for (let x = 0; x < PS; x++) pool[y].push(Math.max(out[2*y][2*x], out[2*y][2*x+1], out[2*y+1][2*x], out[2*y+1][2*x+1])); }
        grid(imgEl, img, { click:true, fixed:true }); grid(outEl, out, {}); grid(poolEl, pool, {}); grid(filtEl, F, {});
        read.replaceChildren(pill('weights in this layer', '3×3 + 1 bias = 10'), pill('feature map size', S + '×' + S), pill('after pooling', PS + '×' + PS), h('span', { class:'pill' }, relu ? 'ReLU zeroed the negative responses: red cells became 0.' : 'Red cells are negative responses, kept because ReLU is off.'));
      }
      draw();
    } };
})();
