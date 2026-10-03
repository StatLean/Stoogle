const $ = (id) => document.getElementById(id);
const node = (tag, text, cls) => { const el = document.createElement(tag); if (text) el.textContent = text; if (cls) el.className = cls; return el; };
async function json(url) { const r = await fetch(url); if (!r.ok) throw new Error(`Request failed (${r.status})`); return r.json(); }
let generation = 0;
function sourceLink(url) {
  const a = node('a', 'View source ↗');
  const parsed = new URL(url);
  if (parsed.protocol === 'https:' && parsed.hostname === 'github.com') { a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; }
  return a;
}
function copyButton(label, text) {
  const b = node('button', label, 'copy'); b.type = 'button';
  b.onclick = async () => { try { await navigator.clipboard.writeText(text); b.textContent = 'Copied'; } catch { b.textContent = 'Copy manually'; } };
  return b;
}
function card(result) {
  const article = node('article', '', 'result');
  article.append(node('h2', result.title || result.name.split('.').pop()));
  article.append(node('div', result.name, 'name'));
  const meta = node('p', '', 'meta'); meta.append(node('span', result.kind, 'badge'), node('span', `${result.module}${result.relevance == null ? '' : ` · relevance ${result.relevance.toFixed(2)}`}`)); article.append(meta);
  if (result.description) article.append(node('p', result.description, 'description'));
  else if (result.generated_description) article.append(node('p', `Generated summary: ${result.generated_description}`, 'description'));
  const details = node('details'); details.append(node('summary', 'Signature and import'));
  let loaded = false;
  details.addEventListener('toggle', async () => {
    if (!details.open || loaded) return;
    loaded = true;
    const content = node('div', 'Loading…'); details.append(content);
    try {
      const d = await json(`/api/declarations/${encodeURIComponent(result.id)}`);
      content.replaceChildren(node('pre', d.import), node('pre', d.signature));
      if (d.name_conflict_modules) content.prepend(node('p', 'This name occurs in several modules; use this result\'s ID and import.', 'error'));
      const actions = node('div', '', 'actions'); actions.append(copyButton('Copy import', d.import), copyButton('Copy name', d.name), sourceLink(d.source_link)); content.append(actions);
      const refs = node('details'); refs.append(node('summary', `Constants in the signature (${d.signature_dependencies.length})`), node('pre', d.signature_dependencies.join('\n') || 'None')); content.append(refs);
      if (d.description) { const prose = node('details'); prose.append(node('summary', 'Full description'), node('pre', d.description)); content.append(prose); }
    } catch (e) { content.textContent = e.message; content.className = 'error'; loaded = false; }
  });
  article.append(details); return article;
}
async function search() {
  const q = $('query').value.trim(); if (!q) return;
  const current = ++generation;
  $('status').className = ''; $('status').textContent = 'Searching…'; $('results').replaceChildren();
  const params = new URLSearchParams({q, limit: '10'});
  if ($('module').value.trim()) params.set('module', $('module').value.trim());
  if ($('kind').value) params.set('kind', $('kind').value);
  history.replaceState(null, '', `?${params}`);
  try {
    const data = await json(`/api/search?${params}`); if (current !== generation) return;
    $('status').textContent = data.count ? `${data.count} candidate${data.count === 1 ? '' : 's'} · ${data.processing_time_ms} ms · ${data.mode}${data.reranked ? ' + reranking' : ''}` : 'No results. Try a shorter name or a broader module filter.';
    if (data.low_confidence) { $('status').textContent += ' · Low relevance: StatLean may not contain this result.'; $('status').className = 'error'; }
    $('results').replaceChildren(...data.results.map(card));
  } catch (e) { if (current === generation) { $('status').textContent = e.message; $('status').className = 'error'; } }
}
$('search-form').onsubmit = (e) => { e.preventDefault(); search(); };
document.querySelectorAll('[data-query]').forEach(b => b.onclick = () => { $('query').value = b.dataset.query; search(); });
json('/api/info').then(info => { $('snapshot').textContent = `${info.declaration_count.toLocaleString('en')} declarations · ${info.mode === 'hybrid' ? 'semantic + keyword search' : 'keyword search'} · StatLean ${info.statlean_tag ? info.statlean_tag + ' / ' : ''}${info.commit.slice(0, 12)}${info.commit_date ? ` (${info.commit_date})` : ''}${info.lean_toolchain ? ` · Lean ${info.lean_toolchain.split(':').pop()}` : ''}`; }).catch(e => { $('snapshot').textContent = e.message; });
const initial = new URLSearchParams(location.search);
if (initial.has('q')) { $('query').value = initial.get('q'); $('module').value = initial.get('module') || ''; $('kind').value = initial.get('kind') || ''; search(); }
