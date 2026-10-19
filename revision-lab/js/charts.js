// Small SVG chart helpers that follow the page's colour tokens.
window.LAB = window.LAB || {};
(function(){
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  LAB.colors = () => ({ fg:css('--fg'), muted:css('--muted'), line:css('--line'), grid:css('--grid'), accent:css('--accent'),
    second:css('--second'), good:css('--good'), bad:css('--bad'), warn:css('--warn'), surface:css('--surface') });
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
  function niceStep(span, count){
    const raw = span / Math.max(1,count), p = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / p;
    return (r <= 1 ? 1 : r <= 2 ? 2 : r <= 5 ? 5 : 10) * p;
  }
  function ticks(min, max, count){
    const st = niceStep(max - min, count), out = [];
    for (let v = Math.ceil(min / st - 1e-9) * st; v <= max + 1e-9; v += st) out.push(+v.toFixed(10));
    return out;
  }
  const fmtDefault = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : +v.toFixed(3) + '');

  // Line chart. series: [{name, data:[[x,y],...], color, dash, points, width}]
  LAB.lineChart = function(el, o){
    const C = LAB.colors(), W = o.w || 560, H = o.h || 250, L = 52, R = 14, T = 16, B = 40;
    let xs = [], ys = [];
    o.series.forEach(s => s.data.forEach(([x,y]) => { if (isFinite(x)&&isFinite(y)) { xs.push(x); ys.push(y); } }));
    const xMin = o.xMin ?? Math.min(...xs), xMax = o.xMax ?? Math.max(...xs);
    let yMin = o.yMin ?? Math.min(...ys), yMax = o.yMax ?? Math.max(...ys);
    if (yMax === yMin) { yMax += 1; yMin -= 1; }
    if (o.yMin == null && o.yMax == null) { const pad = (yMax - yMin) * 0.08; yMin -= pad; yMax += pad; }
    const sx = x => L + (x - xMin) / (xMax - xMin || 1) * (W - L - R);
    const sy = y => T + (1 - (y - yMin) / (yMax - yMin || 1)) * (H - T - B);
    const fx = o.fmtX || fmtDefault, fy = o.fmtY || fmtDefault;
    let g = '';
    ticks(yMin, yMax, o.yTicks || 5).forEach(v => { const y = sy(v);
      g += `<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="${C.grid}"/><text x="${L-6}" y="${y+4}" text-anchor="end" font-size="11" fill="${C.muted}">${esc(fy(v))}</text>`; });
    ticks(xMin, xMax, o.xTicks || 6).forEach(v => { const x = sx(v);
      g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H-B}" stroke="${C.grid}"/><text x="${x}" y="${H-B+16}" text-anchor="middle" font-size="11" fill="${C.muted}">${esc(fx(v))}</text>`; });
    (o.hmarks || []).forEach(m => { const y = sy(m.y);
      g += `<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="${m.color||C.muted}" stroke-dasharray="4 4"/><text x="${W-R-4}" y="${y-4}" text-anchor="end" font-size="11" fill="${m.color||C.muted}">${esc(m.label||'')}</text>`; });
    (o.marks || []).forEach(m => { const x = sx(m.x);
      g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H-B}" stroke="${m.color||C.muted}" stroke-dasharray="4 4"/><text x="${x+4}" y="${T+11}" font-size="11" fill="${m.color||C.muted}">${esc(m.label||'')}</text>`; });
    o.series.forEach(s => {
      const pts = s.data.filter(([x,y]) => isFinite(x)&&isFinite(y));
      const d = pts.map(([x,y],i) => (i?'L':'M') + sx(x).toFixed(1) + ' ' + sy(Math.max(yMin, Math.min(yMax, y))).toFixed(1)).join(' ');
      if (s.area) g += `<path d="${d} L${sx(pts[pts.length-1][0]).toFixed(1)} ${sy(yMin)} L${sx(pts[0][0]).toFixed(1)} ${sy(yMin)} Z" fill="${s.color}" opacity=".12"/>`;
      g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width||2.2}" ${s.dash?`stroke-dasharray="${s.dash}"`:''} stroke-linejoin="round" stroke-linecap="round"/>`;
      if (s.points) pts.forEach(([x,y]) => { g += `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="3.2" fill="${s.color}"/>`; });
    });
    g += `<line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="${C.line}"/><line x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" stroke="${C.line}"/>`;
    if (o.xLabel) g += `<text x="${(L+W-R)/2}" y="${H-4}" text-anchor="middle" font-size="11.5" fill="${C.muted}">${esc(o.xLabel)}</text>`;
    if (o.yLabel) g += `<text transform="translate(12 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="${C.muted}">${esc(o.yLabel)}</text>`;
    if (o.legend !== false) { let lx = L + 8;
      o.series.forEach(s => { if (!s.name) return;
        g += `<line x1="${lx}" x2="${lx+16}" y1="${T+8}" y2="${T+8}" stroke="${s.color}" stroke-width="3" ${s.dash?`stroke-dasharray="${s.dash}"`:''}/><text x="${lx+20}" y="${T+12}" font-size="11.5" fill="${C.fg}">${esc(s.name)}</text>`;
        lx += 28 + s.name.length * 6.6; }); }
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria||'chart')}" font-family="IBM Plex Sans, system-ui, sans-serif">${g}</svg>`;
    return el.firstChild;
  };

  // Bar chart. bars: [{label, value, color}]
  LAB.barChart = function(el, o){
    const C = LAB.colors(), W = o.w || 520, H = o.h || 240, L = 52, R = 14, T = 18, B = 44;
    const vals = o.bars.map(b => b.value);
    let yMin = o.yMin ?? Math.min(0, ...vals), yMax = o.yMax ?? Math.max(...vals) * 1.12;
    if (yMax === yMin) yMax = yMin + 1;
    const sy = y => T + (1 - (y - yMin) / (yMax - yMin)) * (H - T - B);
    const fy = o.fmt || fmtDefault, n = o.bars.length, slot = (W - L - R) / n, bw = Math.min(70, slot * 0.6);
    let g = '';
    ticks(yMin, yMax, 5).forEach(v => { const y = sy(v);
      g += `<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="${C.grid}"/><text x="${L-6}" y="${y+4}" text-anchor="end" font-size="11" fill="${C.muted}">${esc(fy(v))}</text>`; });
    if (o.baseline) { const y = sy(o.baseline.y);
      g += `<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="${C.muted}" stroke-dasharray="4 4"/><text x="${W-R-4}" y="${y-4}" text-anchor="end" font-size="11" fill="${C.muted}">${esc(o.baseline.label)}</text>`; }
    o.bars.forEach((b, i) => { const x = L + slot * i + (slot - bw) / 2, y0 = sy(Math.max(yMin, 0)), y1 = sy(b.value);
      g += `<rect x="${x}" y="${Math.min(y0,y1)}" width="${bw}" height="${Math.abs(y0-y1)}" rx="3" fill="${b.color||C.accent}"/>`;
      g += `<text x="${x+bw/2}" y="${Math.min(y0,y1)-5}" text-anchor="middle" font-size="11.5" fill="${C.fg}" font-family="IBM Plex Mono, monospace">${esc(fy(b.value))}</text>`;
      g += `<text x="${x+bw/2}" y="${H-B+16}" text-anchor="middle" font-size="11.5" fill="${C.muted}">${esc(b.label)}</text>`; });
    g += `<line x1="${L}" x2="${W-R}" y1="${sy(Math.max(yMin,0))}" y2="${sy(Math.max(yMin,0))}" stroke="${C.line}"/>`;
    if (o.yLabel) g += `<text transform="translate(12 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="${C.muted}">${esc(o.yLabel)}</text>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria||'bar chart')}" font-family="IBM Plex Sans, system-ui, sans-serif">${g}</svg>`;
    return el.firstChild;
  };

  // Tiny DOM helpers used by demos and the app.
  LAB.h = function(tag, attrs, ...children){
    const el = document.createElement(tag);
    for (const k in (attrs || {})) {
      if (k === 'class') el.className = attrs[k];
      else if (k === 'html') el.innerHTML = attrs[k];
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] != null) el.setAttribute(k, attrs[k]);
    }
    children.flat().forEach(c => { if (c == null) return; el.append(c.nodeType ? c : document.createTextNode(c)); });
    return el;
  };
  LAB.range = function(label, id, min, max, step, value, fmt){
    const out = LAB.h('output', { id: id + '-out' }, (fmt || (v => v))(value));
    const input = LAB.h('input', { type:'range', id, min, max, step, value });
    const wrap = LAB.h('div', { class:'ctl' }, LAB.h('label', { for:id }, label, out), input);
    input.addEventListener('input', () => { out.textContent = (fmt || (v => v))(+input.value); });
    wrap.input = input; return wrap;
  };
  LAB.seg = function(options, value, onChange){
    const wrap = LAB.h('div', { class:'seg', role:'group' });
    const set = v => { wrap.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.v === v)); wrap.value = v; };
    options.forEach(([v, label]) => wrap.append(LAB.h('button', { type:'button', 'data-v':v, 'aria-pressed': v === value, onclick: () => { set(v); onChange(v); } }, label)));
    wrap.value = value; wrap.set = set; return wrap;
  };
  LAB.store = {
    get(k, d){ try { const v = localStorage.getItem('leyenda:' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v){ try { localStorage.setItem('leyenda:' + k, JSON.stringify(v)); } catch (e) {} }
  };
})();
