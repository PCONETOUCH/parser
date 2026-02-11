window.UI = {
  fmt(v){ return (v===null||v===undefined||Number.isNaN(v)) ? '—' : (typeof v==='number'? v.toLocaleString('ru-RU',{maximumFractionDigits:2}) : v); },
  async loadDashboard(runId){
    const d = await fetch(`/api/dashboard?run_id=${encodeURIComponent(runId||'')}`).then(r=>r.json());
    const k = d.kpi||{};
    const kpi = document.getElementById('kpi');
    const cards = [
      ['Sales proxy', k.sales_proxy],['Sales/day',k.sales_per_day],['Active inventory',k.active_inventory],
      ['Median price/m²',k.median_price_m2],['Quarantine',k.quarantine_count],['Coverage', (k.data_coverage||0)*100+'%']
    ];
    kpi.innerHTML = cards.map(([t,v])=>`<div class='kpi'><div class='t'>${t}</div><div class='v'>${this.fmt(v)}</div></div>`).join('');
    const inx = document.getElementById('insights');
    inx.innerHTML = (d.insights||[]).map(i=>`<div class='card'><b>${i.severity}</b> ${i.title} <a href='/parity'>${i.cta}</a></div>`).join('') || '<div class="text-slate-500">Пока нет инсайтов.</div>';
    window.UICharts.sales('salesChart', (d.charts||{}).sales||[]);
  },
  async loadParity(runId){
    const d = await fetch(`/api/parity?run_id=${encodeURIComponent(runId||'')}`).then(r=>r.json());
    const tb = document.querySelector('#parityTbl tbody');
    tb.innerHTML = (d.rows||[]).map(r=>`<tr><td>${r.complex_name||r.project_name||'UNKNOWN_PROJECT'}</td><td>${r.segment}</td><td>${this.fmt(r.min)}</td><td>${this.fmt(r.max)}</td><td>${this.fmt(r.median)}</td></tr>`).join('');
  },
  async loadProjects(runId){
    const d = await fetch(`/api/parity?run_id=${encodeURIComponent(runId||'')}`).then(r=>r.json());
    const tb = document.querySelector('#projectsTbl tbody');
    tb.innerHTML = (d.rows||[]).map(r=>{
      const p=(r.complex_name||r.project_name||'UNKNOWN_PROJECT');
      return `<tr><td><a href='/projects/${encodeURIComponent(p)}?run_id=${encodeURIComponent(runId||'')}'>${p}</a></td><td>${r.segment}</td><td>${this.fmt(r.median)}</td></tr>`;
    }).join('');
  },
  async loadProjectDetail(runId, project){
    const d = await fetch(`/api/project/${encodeURIComponent(project)}?run_id=${encodeURIComponent(runId||'')}`).then(r=>r.json());
    document.getElementById('projectData').textContent = JSON.stringify(d,null,2);
  },
  async loadRunDetail(runId){
    const d = await fetch(`/api/ops/run/${encodeURIComponent(runId)}`).then(r=>r.json());
    document.getElementById('runJson').textContent = JSON.stringify(d,null,2);
  },
  bindManualAccept(){
    const f = document.getElementById('acceptForm');
    f?.addEventListener('submit', async (e)=>{
      e.preventDefault();
      const payload = Object.fromEntries(new FormData(f).entries());
      const d = await fetch('/api/manual_accept',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).then(r=>r.json());
      document.getElementById('acceptOut').textContent = JSON.stringify(d,null,2);
      alert('Скопировано/Выполнено');
    });
  }
}
