// Interactive benches, part 2: latent space, PCA vs t-SNE, denoising lab, vanishing gradient, LSTM gates, transfer learning, attention.
window.LAB = window.LAB || {};
LAB.demos = LAB.demos || {};
(function(){
  const h = LAB.h, pill = (label, value, cls) => h('span', { class:'pill ' + (cls||'') }, label + ' ', h('b', {}, value));
  function rng(seed){ let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r){ const u = Math.max(1e-9, r()), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  LAB.demos.latent = { title:"Latent space explorer: autoencoder against VAE", blurb:"Two latent numbers and a decoder. Drag the point or move the sliders: z1 morphs the shape from a ring to a stroke, z2 slants it. In VAE mode every position decodes to a believable digit. In plain-autoencoder mode only the islands the training data filled decode cleanly; between them lie holes.",
    render(ctl, stage){
      let z1 = 0.4, z2 = -0.3, mode = 'vae';
      const small = document.createElement('canvas'); small.width = small.height = 28;
      const big = document.createElement('canvas'); big.width = big.height = 168; big.style.width = '168px'; big.style.height = '168px';
      const plane = h('div'), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const r1 = LAB.range('z1: ring ↔ stroke', 'lat-z1', -3, 3, 0.05, z1, v => (+v).toFixed(2)), r2 = LAB.range('z2: slant', 'lat-z2', -3, 3, 0.05, z2, v => (+v).toFixed(2));
      r1.input.oninput = () => { z1 = +r1.input.value; draw(); }; r2.input.oninput = () => { z2 = +r2.input.value; draw(); };
      const seg = LAB.seg([['vae','VAE (KL term on)'],['ae','Plain autoencoder']], mode, v => { mode = v; draw(); });
      ctl.append(seg, r1, r2); stage.append(h('div', { class:'four' }, h('figure', {}, big, h('figcaption', {}, 'decoded 28×28 image')), h('figure', {}, plane, h('figcaption', {}, 'latent plane, click to move the point'))), read, verdict);
      const islands = [[-1.8, 1.4], [1.7, 1.3], [-1.3, -1.6], [1.5, -1.4], [0.1, 0.2]];
      function draw(){
        const C = LAB.colors(), ctx = small.getContext('2d'), t = (z1 + 3) / 6;
        ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 28, 28);
        ctx.save(); ctx.translate(14, 14); ctx.transform(1, 0, z2 * 0.18, 1, 0, 0);
        ctx.beginPath(); ctx.ellipse(0, 0, 9 * (1 - t) + 1.3 * t, 9, 0, 0, Math.PI * 2); ctx.lineWidth = 2.6; ctx.strokeStyle = '#fff'; ctx.stroke(); ctx.restore();
        let hole = 0;
        if (mode === 'ae') { const d = Math.min(...islands.map(([a, b]) => Math.hypot(z1 - a, z2 - b))); hole = Math.max(0, Math.min(1, (d - 0.7) / 1.1));
          if (hole > 0) { const im = ctx.getImageData(0, 0, 28, 28), r = rng(Math.round(z1 * 100) * 7919 + Math.round(z2 * 100)); for (let i = 0; i < im.data.length; i += 4) { const n = r() * 255; im.data[i] = im.data[i] * (1 - hole) + n * hole; im.data[i+1] = im.data[i]; im.data[i+2] = im.data[i]; } ctx.putImageData(im, 0, 0); } }
        const b = big.getContext('2d'); b.imageSmoothingEnabled = false; b.drawImage(small, 0, 0, 168, 168);
        const px = v => 100 + v * 30;
        let dots = ''; if (mode === 'ae') islands.forEach(([a, bb]) => { dots += `<circle cx="${px(a)}" cy="${px(-bb)}" r="22" fill="${C.accent}" opacity=".18"/>`; });
        else dots += `<circle cx="100" cy="100" r="62" fill="${C.accent}" opacity=".10"/><circle cx="100" cy="100" r="32" fill="${C.accent}" opacity=".10"/>`;
        plane.innerHTML = `<svg viewBox="0 0 200 200" width="168" height="168" style="border:1px solid ${C.line};border-radius:6px;background:${C.surface};cursor:crosshair" aria-label="latent plane"><line x1="100" x2="100" y1="0" y2="200" stroke="${C.grid}"/><line x1="0" x2="200" y1="100" y2="100" stroke="${C.grid}"/>${dots}<circle cx="${px(z1)}" cy="${px(-z2)}" r="6" fill="${C.second}"/><text x="4" y="12" font-size="10" fill="${C.muted}">z2</text><text x="186" y="112" font-size="10" fill="${C.muted}">z1</text></svg>`;
        plane.firstChild.onclick = e => { const rect = plane.firstChild.getBoundingClientRect(); z1 = Math.max(-3, Math.min(3, ((e.clientX - rect.left) / rect.width * 200 - 100) / 30)); z2 = Math.max(-3, Math.min(3, -((e.clientY - rect.top) / rect.height * 200 - 100) / 30)); r1.input.value = z1; r2.input.value = z2; r1.input.dispatchEvent(new Event('input')); r2.input.dispatchEvent(new Event('input')); };
        read.replaceChildren(pill('z', '(' + z1.toFixed(2) + ', ' + z2.toFixed(2) + ')'), pill('shape', t < 0.35 ? 'ring, like a 0' : t > 0.7 ? 'stroke, like a 1' : 'in between'), mode === 'ae' ? pill('distance to a trained island', hole === 0 ? 'inside' : 'in a hole', hole ? 'bad' : 'good') : pill('density', 'no holes', 'good'));
        verdict.textContent = mode === 'vae' ? 'The KL term kept every training cloud centred and overlapping, so the decoder has seen codes everywhere near the origin. Any point you choose decodes to a believable digit: the VAE generates.'
          : hole ? 'A plain autoencoder was never asked to keep the codes together. Between the islands the decoder meets a code it has never seen and answers with garbage: a hole. It can only reproduce.' : 'Inside an island the plain autoencoder decodes well: this is a code close to real training images.';
      }
      draw();
    } };

  LAB.demos.embed2d = { title:"PCA against t-SNE on ten classes", blurb:"The same ten clusters drawn two ways. PCA is a straight projection: fast, exact, but the classes overlap. t-SNE keeps neighbours together and gives islands, but the gaps between islands carry no information. Illustrative points, not MNIST.",
    render(ctl, stage){
      let mode = 'pca'; const svg = h('div'), read = h('div', { class:'readout' });
      ctl.append(LAB.seg([['pca','PCA'],['tsne','t-SNE']], mode, v => { mode = v; draw(); })); stage.append(svg, read);
      function draw(){
        const C = LAB.colors(), r = rng(42); let g = '';
        for (let k = 0; k < 10; k++) { const ang = k / 10 * Math.PI * 2, col = `hsl(${k * 36} 60% ${C.fg > '#888' ? 65 : 42}%)`;
          const cx = mode === 'pca' ? 150 + Math.cos(ang) * 32 + Math.sin(ang * 2) * 10 : 150 + Math.cos(ang) * 105, cy = mode === 'pca' ? 150 + Math.sin(ang) * 32 : 150 + Math.sin(ang) * 105;
          for (let i = 0; i < 40; i++) { const a = gauss(r), b = gauss(r); const x = mode === 'pca' ? cx + a * 34 + b * 18 : cx + a * 11, y = mode === 'pca' ? cy + a * 18 + b * 22 : cy + b * 11; g += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.6" fill="${col}" opacity=".8"/>`; } }
        svg.innerHTML = `<svg viewBox="0 0 300 300" style="max-width:360px;border:1px solid ${C.line};border-radius:8px;background:${C.surface}" aria-label="${mode} scatter">${g}<text x="8" y="16" font-size="11" fill="${C.muted}">${mode === 'pca' ? 'first two principal components' : 't-SNE, two dimensions'}</text></svg>`;
        read.replaceChildren(mode === 'pca' ? pill('explained variance of 2 axes', 'about 23% of 32 latent dims') : pill('perplexity', '30 neighbours'), h('span', { class:'pill' }, mode === 'pca' ? 'Use this as classifier input or to check how much spread two axes keep.' : 'Use this to look. Never measure distances between islands, and never feed it to a classifier.'));
      }
      draw();
    } };

  LAB.demos.denoise = { title:"Denoising lab: bottleneck against residual", blurb:"A small synthetic scene with sharp edges and a fine fence. Add Gaussian noise, then compare two cleaners that behave like your two models: A rebuilds the image through a quarter-resolution bottleneck, B removes noise while protecting edges. PSNR is computed live against the clean scene.",
    render(ctl, stage){
      const W = 96; let sigma = 0.02; const clean = new Float32Array(W * W * 3);
      for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { const i = (y * W + x) * 3; let c = [0.45 + 0.35 * y / W, 0.62 + 0.25 * y / W, 0.96];
        if (Math.hypot(x - 72, y - 20) < 8) c = [1, 0.9, 0.45];
        if (y >= 78) c = [0.28, 0.55, 0.26];
        if (x >= 12 && x < 44 && y >= 38 && y < 78) { c = [0.17, 0.17, 0.22]; if ((x - 14) % 6 < 3 && (y - 41) % 8 < 4 && x < 42 && y < 74) c = [0.95, 0.85, 0.45]; }
        if (x >= 54 && x < 90 && y >= 66 && y < 80 && (x % 3 === 0)) c = [0.92, 0.92, 0.9];
        if (x >= 56 && x < 78 && y >= 44 && y < 58) c = y >= 50 && y < 53 ? [1, 1, 1] : [0.85, 0.12, 0.12];
        clean[i] = c[0]; clean[i+1] = c[1]; clean[i+2] = c[2]; }
      const cv = ['clean', 'noisy', 'A: bottleneck', 'B: edge-aware'].map(() => { const c = document.createElement('canvas'); c.width = c.height = W; c.style.width = '192px'; c.style.height = '192px'; return c; });
      const caps = cv.map(() => h('figcaption')), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const r = LAB.range('noise sigma', 'dn-s', 0, 0.15, 0.005, sigma, v => (+v).toFixed(3)); r.input.oninput = () => { sigma = +r.input.value; draw(); };
      const presets = h('div', { class:'seg' }, h('button', { type:'button', onclick: () => { sigma = 0.005; r.input.value = sigma; r.input.dispatchEvent(new Event('input')); } }, 'TouNum level, 0.005'), h('button', { type:'button', onclick: () => { sigma = 0.1; r.input.value = sigma; r.input.dispatchEvent(new Event('input')); } }, 'Heavy, 0.10'));
      ctl.append(r, h('div', { class:'ctl' }, h('label', {}, 'presets'), presets)); stage.append(h('div', { class:'four' }, ...cv.map((c, i) => h('figure', {}, c, caps[i]))), read, verdict);
      const show = (c, buf) => { const ctx = c.getContext('2d'), im = ctx.createImageData(W, W); for (let p = 0, q = 0; p < buf.length; p += 3, q += 4) { im.data[q] = buf[p] * 255; im.data[q+1] = buf[p+1] * 255; im.data[q+2] = buf[p+2] * 255; im.data[q+3] = 255; } ctx.putImageData(im, 0, 0); };
      const psnr = buf => { let s = 0; for (let i = 0; i < buf.length; i++) s += (buf[i] - clean[i]) ** 2; const mse = s / buf.length; return mse < 1e-10 ? 99 : 10 * Math.log10(1 / mse); };
      function modelA(src){ const S = W / 4, small = new Float32Array(S * S * 3), out = new Float32Array(src.length);
        for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) for (let c = 0; c < 3; c++) { let s = 0; for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) s += src[((y*4+j) * W + x*4+i) * 3 + c]; small[(y * S + x) * 3 + c] = s / 16; }
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { const fy = Math.min(S - 1, Math.max(0, (y + 0.5) / 4 - 0.5)), fx = Math.min(S - 1, Math.max(0, (x + 0.5) / 4 - 0.5)), y0 = Math.floor(fy), x0 = Math.floor(fx), y1 = Math.min(S-1, y0+1), x1 = Math.min(S-1, x0+1), ty = fy - y0, tx = fx - x0;
          for (let c = 0; c < 3; c++) { const v = small[(y0*S+x0)*3+c]*(1-ty)*(1-tx) + small[(y0*S+x1)*3+c]*(1-ty)*tx + small[(y1*S+x0)*3+c]*ty*(1-tx) + small[(y1*S+x1)*3+c]*ty*tx; out[(y*W+x)*3+c] = v; } }
        return out; }
      function modelB(src, sg){ const out = new Float32Array(src.length), sr = Math.max(0.03, 2.4 * sg), sp = [];
        for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) sp.push([j, i, Math.exp(-(i*i + j*j) / (2 * 1.6 * 1.6))]);
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { const p = (y * W + x) * 3; let ws = 0, acc = [0, 0, 0];
          for (const [j, i, w0] of sp) { const yy = Math.min(W-1, Math.max(0, y + j)), xx = Math.min(W-1, Math.max(0, x + i)), q = (yy * W + xx) * 3; const d = (src[q]-src[p])**2 + (src[q+1]-src[p+1])**2 + (src[q+2]-src[p+2])**2; const w = w0 * Math.exp(-d / (2 * sr * sr * 3)); ws += w; acc[0] += w * src[q]; acc[1] += w * src[q+1]; acc[2] += w * src[q+2]; }
          out[p] = acc[0] / ws; out[p+1] = acc[1] / ws; out[p+2] = acc[2] / ws; }
        return out; }
      function draw(){
        const r0 = rng(7), noisy = new Float32Array(clean.length); for (let i = 0; i < noisy.length; i++) noisy[i] = Math.min(1, Math.max(0, clean[i] + sigma * gauss(r0)));
        const a = modelA(noisy), b = modelB(noisy, sigma), pn = psnr(noisy), pa = psnr(a), pb = psnr(b);
        [clean, noisy, a, b].forEach((buf, i) => show(cv[i], buf));
        caps[0].innerHTML = 'clean target'; caps[1].innerHTML = `noisy · <b>${pn.toFixed(2)} dB</b>`; caps[2].innerHTML = `A, bottleneck · <b>${pa.toFixed(2)} dB</b>`; caps[3].innerHTML = `B, edge-aware · <b>${pb.toFixed(2)} dB</b>`;
        read.replaceChildren(pill('gain A', (pa - pn >= 0 ? '+' : '') + (pa - pn).toFixed(2) + ' dB', pa - pn >= 0 ? 'good' : 'bad'), pill('gain B', (pb - pn >= 0 ? '+' : '') + (pb - pn).toFixed(2) + ' dB', pb - pn >= 0 ? 'good' : 'bad'), pill('your Livrable 2', 'A −2.93 · B +4.18'));
        verdict.textContent = sigma <= 0.02 ? 'Mild noise, the TouNum case: the bottleneck blurs the fence and the windows more than the noise hurt them, so A loses decibels. B only touches flat areas and gains. This is the whole story of Model A against Model B.'
          : sigma < 0.08 ? 'Medium noise: both help, but A pays for the blur on every edge while B keeps them, so B stays ahead.' : 'Heavy noise: now even a blur gains, because the noise costs more than the lost detail. The ranking is the same, the margin smaller.';
      }
      draw();
    } };

  LAB.demos.vanishing = { title:"The vanishing gradient", blurb:"In a simple RNN the training signal that reaches word 1 from word 30 is multiplied by about the same factor at every step. Slide that factor. The LSTM cell state is updated by addition, so its path keeps the signal near 1.",
    render(ctl, stage){
      let g = 0.85; const chart = h('div'), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const r = LAB.range('factor per time step', 'vg-g', 0.5, 1.5, 0.01, g, v => (+v).toFixed(2)); r.input.oninput = () => { g = +r.input.value; draw(); };
      ctl.append(r); stage.append(chart, read, verdict);
      function draw(){
        const C = LAB.colors(), d = [], l = []; for (let t = 0; t <= 30; t++) { d.push([t, Math.log10(g ** t)]); l.push([t, Math.log10(0.97 + 0.03 * Math.cos(t))]); }
        LAB.lineChart(chart, { h:230, yMin:-6, yMax:6, series:[{ name:'simple RNN: factor^t', data:d, color:C.second }, { name:'LSTM cell path', data:l, color:C.accent }], xLabel:'words back in time', yLabel:'gradient size, log10', fmtY: v => '10^' + v, aria:'gradient over time' });
        const v30 = g ** 30;
        read.replaceChildren(pill('after 30 words', v30 < 1e-3 || v30 > 1e3 ? v30.toExponential(1) : v30.toFixed(3), g < 0.95 || g > 1.05 ? 'bad' : 'good'));
        verdict.textContent = g < 0.95 ? 'Vanishing: the first words of the sentence have no influence left on the loss, so the network cannot learn long dependencies. Sigmoid and tanh slopes below 1 make this the common case.' : g > 1.05 ? 'Exploding: the gradient grows without limit and the weights jump. Gradient clipping is the usual guard.' : 'Near 1 the signal survives. Gates in LSTM and GRU are what hold the factor near 1 on the memory path.';
      }
      draw();
    } };

  LAB.demos.lstm = { title:"LSTM gate simulator", blurb:"One time step. The forget gate scales the old cell state, the input gate scales the new candidate, their sum is the new cell state, and the output gate decides how much of it becomes the hidden state. Every gate is a sigmoid between 0 and 1.",
    render(ctl, stage){
      let f = 0.9, i = 0.2, o = 0.8, cand = 0.6, cOld = -0.4; const svg = h('div'), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const mk = (lab, id, min, max, val, set) => { const r = LAB.range(lab, id, min, max, 0.01, val, v => (+v).toFixed(2)); r.input.oninput = () => { set(+r.input.value); draw(); }; return r; };
      ctl.append(mk('forget gate f', 'ls-f', 0, 1, f, v => f = v), mk('input gate i', 'ls-i', 0, 1, i, v => i = v), mk('candidate value c̃', 'ls-c', -1, 1, cand, v => cand = v), mk('old cell state C(t−1)', 'ls-co', -1, 1, cOld, v => cOld = v), mk('output gate o', 'ls-o', 0, 1, o, v => o = v));
      stage.append(svg, read, verdict);
      function draw(){
        const C = LAB.colors(), cNew = f * cOld + i * cand, hNew = o * Math.tanh(cNew), box = (x, y, w, t, sub) => `<rect x="${x}" y="${y}" width="${w}" height="44" rx="7" fill="${C.surface}" stroke="${C.line}"/><text x="${x+w/2}" y="${y+18}" text-anchor="middle" font-size="11.5" fill="${C.muted}">${t}</text><text x="${x+w/2}" y="${y+35}" text-anchor="middle" font-size="13" font-family="IBM Plex Mono, monospace" fill="${C.fg}">${sub}</text>`;
        const arrow = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${C.muted}" stroke-width="1.6" marker-end="url(#ah)"/>`;
        svg.innerHTML = `<svg viewBox="0 0 560 200" aria-label="LSTM cell"><defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="${C.muted}"/></marker></defs>
          ${box(10, 20, 110, 'old cell C(t−1)', cOld.toFixed(2))}${box(150, 20, 90, '× forget f', (f * cOld).toFixed(2))}${box(270, 20, 70, '+', cNew.toFixed(2))}${box(370, 20, 80, 'tanh', Math.tanh(cNew).toFixed(2))}${box(470, 20, 80, '× output o', hNew.toFixed(2))}
          ${box(150, 130, 90, 'candidate c̃', cand.toFixed(2))}${box(270, 130, 70, '× input i', (i * cand).toFixed(2))}
          ${arrow(120, 42, 148, 42)}${arrow(240, 42, 268, 42)}${arrow(340, 42, 368, 42)}${arrow(450, 42, 468, 42)}${arrow(240, 152, 268, 152)}${arrow(305, 130, 305, 66)}
          <text x="305" y="92" text-anchor="middle" font-size="11" fill="${C.accent}">new cell C(t) = ${cNew.toFixed(2)}</text><text x="510" y="82" text-anchor="middle" font-size="11" fill="${C.accent}">hidden h(t) = ${hNew.toFixed(2)}</text></svg>`;
        read.replaceChildren(pill('C(t) = f·C(t−1) + i·c̃', `${f.toFixed(2)}·${cOld.toFixed(2)} + ${i.toFixed(2)}·${cand.toFixed(2)} = ${cNew.toFixed(2)}`), pill('h(t) = o·tanh(C(t))', hNew.toFixed(2)));
        verdict.textContent = f > 0.8 && i < 0.3 ? 'Forget near 1, input near 0: the belt is carried along almost unchanged. This is how a memory survives thirty words: the update is additive and the gate stays open.' : f < 0.2 ? 'Forget near 0: the old memory is erased, and the cell is rewritten from the candidate. Useful at the start of a new clause.' : i > 0.7 ? 'Input near 1: the candidate is written strongly onto the belt, mixed with what the forget gate kept.' : 'A mix: part of the old memory kept, part of the candidate written. Training learns these gate values from the words themselves.';
      }
      draw();
    } };

  LAB.demos.transfer = { title:"From scratch, frozen features, or fine-tuned", blurb:"Three ways to train a cat-or-dog classifier, drawn as learning curves. Slide the dataset size. The pretrained body gives a strong start regardless of size; training from scratch only catches up with lots of images; fine-tuning helps once there is enough data not to over-fit. Illustrative curves.",
    render(ctl, stage){
      let n = 2000; const chart = h('div'), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const r = LAB.range('training images', 'tl-n', 2, 4.3, 0.01, Math.log10(n), v => Math.round(10 ** v).toLocaleString()); r.input.oninput = () => { n = Math.round(10 ** r.input.value); draw(); };
      ctl.append(r); stage.append(chart, read, verdict);
      function draw(){
        const C = LAB.colors(), L = (Math.log10(n) - 2) / 2.3, s = [], fz = [], ft = [];
        const fS = 0.55 + 0.4 * L, fZ = 0.90, fT = 0.86 + 0.08 * L;
        for (let e = 1; e <= 30; e++) { s.push([e, 0.5 + (fS - 0.5) * (1 - Math.exp(-0.12 * e))]); fz.push([e, 0.5 + (fZ - 0.5) * (1 - Math.exp(-0.5 * e))]); ft.push([e, 0.5 + (fT - 0.5) * (1 - Math.exp(-0.3 * e))]); }
        LAB.lineChart(chart, { h:240, yMin:0.45, yMax:1, series:[{ name:'from scratch', data:s, color:C.bad }, { name:'frozen InceptionV3 + head', data:fz, color:C.accent }, { name:'fine-tuned top blocks, lr 1e-5', data:ft, color:C.second }], xLabel:'epoch', yLabel:'validation accuracy', fmtY: v => Math.round(v * 100) + '%', aria:'transfer learning curves' });
        read.replaceChildren(pill('scratch after 30 epochs', Math.round(fS * 100) + '%'), pill('frozen', Math.round(fZ * 100) + '%'), pill('fine-tuned', Math.round(fT * 100) + '%', fT > fZ ? 'good' : 'warn'));
        verdict.textContent = n < 800 ? 'Few images: freeze the body. Fine-tuning has too little data and over-fits; from scratch barely learns.' : n < 5000 ? 'A few thousand images, the usual case: frozen features already reach about 90%, and careful fine-tuning adds a little.' : 'Many images: fine-tuning wins, and even from scratch becomes competitive, at a far higher cost in time.';
      }
      draw();
    } };

  LAB.demos.attention = { title:"Attention over 64 regions", blurb:"An 8×8 grid stands for the 64 InceptionV3 regions of a photo. Click each word of the caption: the decoder's hidden state scores all regions, softmax turns the scores into weights that sum to 1, and the bright cells are where the context vector comes from.",
    render(ctl, stage){
      const words = ['a', 'man', 'riding', 'a', 'horse', 'on', 'grass', '<end>'];
      const region = (y, x) => (x >= 5 && x <= 6 && y >= 2 && y <= 5) ? 'man' : (y >= 4 && y <= 6 && x >= 1 && x <= 5) ? 'horse' : y >= 6 ? 'grass' : (y >= 1 && y <= 3 && x <= 2) ? 'trees' : 'sky';
      const colors = { sky:'#9ec9ea', trees:'#2f6b3a', horse:'#8a5a2b', man:'#3d5fb3', grass:'#6aa84f' };
      const scores = { 0:{ man:1.2, horse:0.6 }, 1:{ man:3.2, horse:0.6 }, 2:{ man:2.0, horse:2.3 }, 3:{ horse:1.6 }, 4:{ horse:3.4, man:0.4 }, 5:{ horse:1.2, grass:1.9 }, 6:{ grass:3.1 }, 7:{ sky:0.8, grass:0.6 } };
      let wi = 4; const grid = h('div', { class:'grid-cells' }), read = h('div', { class:'readout' }), verdict = h('div', { class:'verdict' });
      const wordRow = h('div', { class:'words' }, ...words.map((w, k) => h('button', { type:'button', 'aria-pressed': k === wi, onclick: () => { wi = k; wordRow.querySelectorAll('button').forEach((b, j) => b.setAttribute('aria-pressed', j === k)); draw(); } }, w)));
      ctl.append(h('div', { class:'ctl' }, h('label', {}, 'generated caption, click a word'), wordRow)); stage.append(grid, read, verdict); grid.style.gridTemplateColumns = 'repeat(8, 30px)';
      function draw(){
        const r = rng(100 + wi), sc = [], sum = {}; let m = -1e9;
        for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { const s = (scores[wi][region(y, x)] || 0) + (r() - 0.5) * 0.7; sc.push(s); m = Math.max(m, s); }
        const ex = sc.map(s => Math.exp(s - m)), Z = ex.reduce((a, b) => a + b, 0), w = ex.map(e => e / Z), wMax = Math.max(...w);
        grid.replaceChildren(...w.map((v, k) => { const y = Math.floor(k / 8), x = k % 8, reg = region(y, x); sum[reg] = (sum[reg] || 0) + v; const c = h('div', { class:'cell', title: reg + ' ' + v.toFixed(3) }, (v * 100).toFixed(0)); c.style.background = colors[reg]; c.style.color = '#fff'; c.style.boxShadow = `inset 0 0 0 30px rgba(255,255,255,${(0.75 * v / wMax).toFixed(2)})`; c.style.color = v / wMax > 0.5 ? '#111' : '#fff'; return c; }));
        const top = Object.entries(sum).sort((a, b) => b[1] - a[1]);
        read.replaceChildren(pill('word', words[wi]), pill('sum of 64 weights', w.reduce((a, b) => a + b, 0).toFixed(2), 'good'), ...top.slice(0, 3).map(([k, v]) => pill(k, Math.round(v * 100) + '%')));
        verdict.textContent = `context vector = ${top.slice(0, 2).map(([k, v]) => `${v.toFixed(2)} × ${k} features`).join(' + ')} + the rest. The decoder reads this vector together with the embedding of the previous word, then predicts the next one.`;
      }
      draw();
    } };
})();
