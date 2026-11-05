/* ==========================================================
   CancerPredict — frontend logic
   The forms are rendered from data (no duplicated markup), so
   adding a question means editing ONE list below.
   ========================================================== */

const API = {
  breast:   '/api/predict/breast',
  prostate: '/api/predict/prostate',
};

/* ---------------- assessment definitions ---------------- */
const ASSESSMENTS = {
  breast: {
    title: 'Breast cancer risk assessment',
    sub: 'Ten questions about your lifestyle and reproductive history.',
    model: 'Logistic Regression · k=10 features',
    icon: '<path d="M12 21s-7-4.4-7-9.5a4 4 0 0 1 7-2.6 4 4 0 0 1 7 2.6C19 16.6 12 21 12 21z"/>',
    fields: [
      { key: 'age', label: 'Your age', type: 'number', min: 20, max: 90, placeholder: 'e.g. 45', unit: 'years' },
      { key: 'menarche', label: 'Age at first menstrual period', type: 'number', min: 8, max: 17, placeholder: 'e.g. 13', unit: 'years' },
      { key: 'children', label: 'Number of children', type: 'number', min: 0, max: 5, placeholder: 'e.g. 2', unit: 'enter 5 if five or more', reveal: 'child' },
      { key: 'family_history', label: 'A close relative diagnosed with breast cancer?', type: 'bool' },
      { key: 'hyperplasia', label: 'Ever diagnosed with breast hyperplasia?', type: 'bool' },
      { key: 'tobacco', label: 'Do you use tobacco?', type: 'bool' },
      { key: 'alcohol', label: 'Do you drink alcohol?', type: 'bool' },
      { key: 'is_menopausal', label: 'Have you reached menopause?', type: 'bool', reveal: 'meno' },
      { key: 'menopause_age', label: 'Age at menopause', type: 'number', min: 30, max: 65, placeholder: 'e.g. 50', unit: 'years', slot: 'meno' },
      { key: 'age_first_birth', label: 'Age at first childbirth', type: 'number', min: 12, max: 50, placeholder: 'e.g. 28', unit: 'years', slot: 'child' },
      { key: 'breastfeeding_months', label: 'Total months breastfeeding', type: 'number', min: 0, max: 72, placeholder: 'e.g. 12', unit: 'across all children', slot: 'child' },
    ],
  },

  prostate: {
    title: 'Prostate cancer risk assessment',
    sub: 'Six clinical values. Four come from a routine PSA blood panel.',
    model: 'Logistic Regression · k=6 features',
    icon: '<path d="M4 20V10M10 20V4M16 20v-8M22 20H2"/>',
    fields: [
      { key: 'age', label: 'Your age', type: 'number', min: 40, max: 90, placeholder: 'e.g. 63', unit: 'years' },
      { key: 'bmi', label: 'Body mass index', type: 'number', min: 15, max: 45, step: 0.1, placeholder: 'e.g. 27.0', unit: 'kg/m²' },
      { key: 'psa', label: 'PSA level', type: 'number', min: 0, max: 50, step: 0.1, placeholder: 'e.g. 5.0', unit: 'ng/mL — from a blood test' },
      { key: 'free_psa_ratio', label: 'Free PSA ratio', type: 'number', min: 0, max: 1, step: 0.01, placeholder: 'e.g. 0.20', unit: 'free PSA ÷ total PSA' },
      { key: 'testosterone', label: 'Testosterone level', type: 'number', min: 100, max: 1000, placeholder: 'e.g. 480', unit: 'ng/dL' },
      { key: 'alp', label: 'Alkaline phosphatase', type: 'number', min: 20, max: 200, step: 0.1, placeholder: 'e.g. 98.0', unit: 'U/L' },
    ],
  },
};

/* fields that live in a collapsible 'slot' (shown conditionally) */
const SLOT_LABELS = { meno: 'Menopause details', child: 'Childbearing details' };

/* landing page content — also rendered from data */
const PICKS = [
  {
    type: 'breast',
    tag: '10 questions',
    icon: '<path d="M12 21s-7-4.4-7-9.5a4 4 0 0 1 7-2.6 4 4 0 0 1 7 2.6C19 16.6 12 21 12 21z"/>',
    title: 'Breast screening',
    desc: 'Lifestyle and reproductive history mapped to ten clinical features. Our strongest model — 99% sensitivity.',
  },
  {
    type: 'prostate',
    tag: '6 values',
    icon: '<path d="M4 20V10M10 20V4M16 20v-8M22 20H2"/>',
    title: 'Prostate screening',
    desc: 'Age, BMI and a routine PSA blood panel — the same inputs a clinician uses during screening.',
  },
  {
    type: null,
    tag: 'Both',
    icon: '<path d="M12 3l7.5 3.4v5.3c0 4.4-3 7.6-7.5 8.8-4.5-1.2-7.5-4.4-7.5-8.8V6.4z"/><path d="M9 12l2 2 4-4"/>',
    title: 'Compare both',
    desc: 'Run each assessment in turn and see which factors move your risk estimate the most.',
  },
];

const RESULTS = [
  {
    name: 'Breast · Logistic Regression',
    badge: 'Deployed',
    rows: [['Accuracy', '0.93'], ['Recall', '0.99', true], ['F1 score', '0.95'], ['ROC-AUC', '0.96']],
  },
  {
    name: 'Prostate · Logistic Regression',
    badge: 'Deployed',
    rows: [['Accuracy', '0.72'], ['Recall', '0.71', true], ['F1 score', '0.72'], ['ROC-AUC', '0.72']],
  },
  {
    name: 'Baselines (KNN / SVM)',
    badge: 'Benchmark',
    rows: [['Breast ROC-AUC', '0.93–0.95'], ['Prostate ROC-AUC', '0.73'], ['CV folds', '5-fold stratified'], ['Resampling', 'SMOTE']],
  },
];

/* ---------------- state ---------------- */
let currentType = 'breast';
let lastResult = null;
let openSlots = {};

/* ---------------- navigation ---------------- */
function go(screen, push = true) {
  // The assessment screen is built from data, so make sure it exists
  if (screen === 'assessment') return startAssessment(currentType, push);

  document.querySelectorAll('.screen').forEach(s => s.classList.remove('screen--on'));
  document.getElementById(screen).classList.add('screen--on');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  // Push a real history entry so browser back/forward work
  if (push) {
    history.pushState({ screen }, '', '#' + screen);
  }
  if (screen === 'landing') renderLanding();
}

// Handle browser back / forward
window.addEventListener('popstate', (e) => {
  const hash = location.hash.replace('#', '');
  const screen = (e.state && e.state.screen) || hash || 'landing';
  if (screen === 'assessment') {
    startAssessment(currentType, false);  // rebuild form without pushing again
  } else if (screen === 'landing' || screen === 'result') {
    go(screen, false);
  } else {
    // an in-page anchor (#method / #results) — go back to landing and scroll
    go('landing', false);
    const target = document.getElementById(screen);
    if (target) setTimeout(() => target.scrollIntoView({ behavior: 'smooth' }), 40);
  }
});

/* scroll to a section on the landing page (from the nav links) */
function jumpTo(id) {
  const landing = document.getElementById('landing');
  if (!landing.classList.contains('screen--on')) go('landing', false);
  const target = document.getElementById(id);
  if (target) setTimeout(() => target.scrollIntoView({ behavior: 'smooth' }), 40);
}

/* ---------------- landing content ---------------- */
function renderLanding() {
  const row = document.getElementById('pickerRow');
  row.innerHTML = PICKS.map(p => `
    <button class="pick" data-type="${p.type || ''}">
      <span class="pick__tag">${p.tag}</span>
      <span class="pick__ico"><svg viewBox="0 0 24 24">${p.icon}</svg></span>
      <h3>${p.title}</h3>
      <p>${p.desc}</p>
      <span class="pick__go">${p.type ? 'Start assessment' : 'Learn more'}
        <svg viewBox="0 0 24 24"><path d="M5 12h13M13 6l6 6-6 6"/></svg>
      </span>
    </button>`).join('');

  row.querySelectorAll('.pick').forEach(el => {
    el.addEventListener('click', () => {
      const t = el.dataset.type;
      if (!t) { document.getElementById('method').scrollIntoView({ behavior: 'smooth' }); return; }
      startAssessment(t);
    });
  });

  const grid = document.getElementById('resultsGrid');
  grid.innerHTML = RESULTS.map(r => `
    <div class="res">
      <div class="res__head"><h3>${r.name}</h3><span class="res__badge">${r.badge}</span></div>
      ${r.rows.map(([k, v, hi]) => `
        <div class="res__row ${hi ? 'res__row--hi' : ''}"><span>${k}</span><b>${v}</b></div>`).join('')}
    </div>`).join('');
}

/* ---------------- assessment ---------------- */
function startAssessment(type, push = true) {
  currentType = type;
  openSlots = {};
  const a = ASSESSMENTS[type];

  document.getElementById('seg').innerHTML = Object.keys(ASSESSMENTS).map(k =>
    `<button data-type="${k}" class="${k === type ? 'is-on' : ''}">${k === 'breast' ? 'Breast' : 'Prostate'}</button>`
  ).join('');
  document.querySelectorAll('#seg button').forEach(b =>
    b.addEventListener('click', () => startAssessment(b.dataset.type, false)));

  document.getElementById('sheetTitle').textContent = a.title;
  document.getElementById('sheetSub').textContent = a.sub;

  const holder = document.getElementById('fields');
  holder.innerHTML = '';

  // main fields (no slot)
  a.fields.filter(f => !f.slot).forEach(f => holder.appendChild(buildField(f, a)));

  // conditional slots
  Object.keys(SLOT_LABELS).forEach(slot => {
    const slotFields = a.fields.filter(f => f.slot === slot);
    if (!slotFields.length) return;

    const wrap = document.createElement('div');
    wrap.className = 'q q--reveal';
    wrap.id = 'slot-' + slot;
    wrap.innerHTML = `<div><div class="q--split">${slotFields.map(f => buildField(f, a).outerHTML).join('')}</div></div>`;
    holder.appendChild(wrap);
  });

  // // listeners for conditional reveals
  // holder.querySelectorAll('input').forEach(el => {
  //   el.addEventListener('input', () => syncSlots(a));
  //   el.addEventListener('change', () => syncSlots(a));
  // });
    // listeners: conditional reveals + clear the field's error once answered
  holder.querySelectorAll('input').forEach(el => {
    const onAnswer = () => { clearError(el.name); syncSlots(a); };
    el.addEventListener('input', onAnswer);
    el.addEventListener('change', onAnswer);
  });

  syncSlots(a);

  // show the screen ourselves (go('assessment') would call back into us)
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('screen--on'));
  document.getElementById('assessment').classList.add('screen--on');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (push) history.pushState({ screen: 'assessment' }, '', '#assessment');
}

/* build a single question block */
function buildField(f) {
  const wrap = document.createElement('div');
  wrap.className = 'q';

  if (f.type === 'bool') {
    wrap.innerHTML = `
      <span class="q__label">${f.label}</span>
      <div class="toggle" role="radiogroup" aria-label="${f.label}">
        <label class="toggle__opt">
          <input type="radio" name="${f.key}" value="0">
          <span>No</span>
        </label>
        <label class="toggle__opt">
          <input type="radio" name="${f.key}" value="1">
          <span>Yes</span>
        </label>
      </div>
      <p class="q__error" data-error-for="${f.key}" role="alert" hidden></p>`;
  } else {
    wrap.innerHTML = `
      <label class="q__label" for="${f.key}">${f.label}</label>
      <input type="number" id="${f.key}" name="${f.key}"
             min="${f.min ?? ''}" max="${f.max ?? ''}" step="${f.step ?? 1}"
             placeholder="${f.placeholder ?? ''}">
      ${f.unit ? `<span class="q__hint">${f.unit}</span>` : ''}
      <p class="q__error" data-error-for="${f.key}" role="alert" hidden></p>`;
  }
  return wrap;
}

// /* build a single question block */
// function buildField(f) {
//   const wrap = document.createElement('div');
//   wrap.className = 'q';

//   if (f.type === 'bool') {
//     // label wraps input+span so clicking the span always toggles the input
//     wrap.innerHTML = `
//       <span class="q__label">${f.label}</span>
//       <div class="toggle" role="group">
//         <label class="toggle__opt">
//           <input type="radio" name="${f.key}" value="0" checked>
//           <span>No</span>
//         </label>
//         <label class="toggle__opt">
//           <input type="radio" name="${f.key}" value="1">
//           <span>Yes</span>
//         </label>
//       </div>`;
//   } else {
//     wrap.innerHTML = `
//       <label class="q__label" for="${f.key}">${f.label}</label>
//       <input type="number" id="${f.key}" name="${f.key}"
//              min="${f.min ?? ''}" max="${f.max ?? ''}" step="${f.step ?? 1}"
//              placeholder="${f.placeholder ?? ''}">
//       ${f.unit ? `<span class="q__hint">${f.unit}</span>` : ''}`;
//   }
//   return wrap;
// }

/* show/hide conditional slots based on current answers */
function syncSlots(a) {
  const { data: answers } = collect();
  const rules = {
    meno: answers.is_menopausal === true,
    child: (answers.children || 0) > 0,
  };
  Object.keys(rules).forEach(slot => {
    const el = document.getElementById('slot-' + slot);
    if (!el) return;
    const shouldOpen = rules[slot];
    if (shouldOpen !== openSlots[slot]) {
      openSlots[slot] = shouldOpen;
      el.classList.toggle('is-open', shouldOpen);
      if (shouldOpen) {
        el.querySelectorAll('input').forEach(i => { if (i.type === 'radio') i.disabled = false; });
      }
    }
  });
}

/* read all current answers into a plain object.
   Returns { data, missing } — missing lists any required empty number fields. */
function collect() {
  const a = ASSESSMENTS[currentType];
  const out = {};
  const missing = [];
  a.fields.forEach(f => {
    if (f.slot) {
      // slot fields only count when their slot is open
      const slotOpen = document.getElementById('slot-' + f.slot)?.classList.contains('is-open');
      if (!slotOpen) { out[f.key] = (f.type === 'bool') ? false : 0; return; }
    }
    if (f.type === 'bool') {
      const checked = document.querySelector(`input[name="${f.key}"]:checked`);
      out[f.key] = checked ? checked.value === '1' : false;
    } else {
      const el = document.getElementById(f.key);
      if (!el) return;
      if (el.value === '') { missing.push(f.label); out[f.key] = 0; }
      else out[f.key] = Number(el.value);
    }
  });
  return { data: out, missing };
}
/* ---------------- validation ---------------- */
function setError(key, msg) {
  const p = document.querySelector(`[data-error-for="${key}"]`);
  if (!p) return;
  p.textContent = msg;
  p.hidden = !msg;
  const q = p.closest('.q');
  q.classList.toggle('q--error', !!msg);
  const num = document.getElementById(key);
  if (num) num.setAttribute('aria-invalid', msg ? 'true' : 'false');
  if (msg) {                       // restart the shake animation
    q.classList.remove('q--shake'); void q.offsetWidth; q.classList.add('q--shake');
  }
}

function clearError(key) { setError(key, ''); }

/* validates every visible field; returns the key of the first invalid one (or null) */
function validateForm() {
  const a = ASSESSMENTS[currentType];
  let firstBad = null;

  a.fields.forEach(f => {
    // fields inside a closed slot aren't asked, so they aren't required
    if (f.slot && !document.getElementById('slot-' + f.slot)?.classList.contains('is-open')) {
      clearError(f.key);
      return;
    }

    let msg = '';
    if (f.type === 'bool') {
      if (!document.querySelector(`input[name="${f.key}"]:checked`)) {
        msg = 'Please choose Yes or No.';
      }
    } else {
      const el = document.getElementById(f.key);
      if (!el) return;
      if (el.value === '') {
        msg = 'Please enter a value.';
      } else {
        const n = Number(el.value);
        if ((f.min != null && n < f.min) || (f.max != null && n > f.max)) {
          msg = `Enter a value between ${f.min} and ${f.max}.`;
        }
      }
    }

    setError(f.key, msg);
    if (msg && !firstBad) firstBad = f.key;
  });

  return firstBad;
}

function focusField(key) {
  const el = document.getElementById(key) || document.querySelector(`input[name="${key}"]`);
  if (!el) return;
  el.closest('.q').scrollIntoView({ behavior: 'smooth', block: 'center' });
  el.focus({ preventScroll: true });
}
// submit...................
document.getElementById('form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const btn = ev.target.querySelector('button[type="submit"]');

  const firstBad = validateForm();
  if (firstBad) { focusField(firstBad); return; }

  const { data } = collect();

  const label = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = 'Calculating…';

  try {
    const res = await fetch(API[currentType], {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error((await res.text()) || ('HTTP ' + res.status));
    lastResult = await res.json();
    renderResult(lastResult, currentType);
    go('result');
  } catch (err) {
    alert('Something went wrong: ' + err.message);
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = label;
  }
});
// /* ---------------- submit ---------------- */
// document.getElementById('form').addEventListener('submit', async (ev) => {
//   ev.preventDefault();
//   const btn = ev.target.querySelector('button[type="submit"]');

//   // const { data, missing } = collect();
//   // if (missing.length) {
//   //   alert('Please fill in: ' + missing.join(', '));
//   //   return;
//   // }
//     const firstBad = validateForm();
//   if (firstBad) { focusField(firstBad); return; }

//   const { data } = collect();

//   const label = btn.innerHTML;
//   btn.disabled = true;
//   btn.innerHTML = 'Calculating…';

//   try {
//     const res = await fetch(API[currentType], {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify(data),
//     });
//     if (!res.ok) throw new Error((await res.text()) || ('HTTP ' + res.status));
//     lastResult = await res.json();
//     renderResult(lastResult, currentType);
//     go('result');
//   } catch (err) {
//     alert('Something went wrong: ' + err.message);
//     console.error(err);
//   } finally {
//     btn.disabled = false;
//     btn.innerHTML = label;
//   }
// });

/* ---------------- result ---------------- */
const VERDICTS = {
  lower:    ['Lower estimated risk',    'var(--ok)'],
  elevated: ['Elevated estimated risk', 'var(--warn)'],
  higher:   ['Higher estimated risk',   'var(--bad)'],
};

function renderResult(r, type) {
  const pct = Math.round(r.risk * 100);
  const a = ASSESSMENTS[type];

  document.getElementById('rType').textContent = a.title;
  const [label, colour] = VERDICTS[r.verdict] || VERDICTS.lower;
  const v = document.getElementById('rVerdict');
  v.textContent = label;
  v.style.color = colour;

  document.getElementById('rMessage').textContent = r.message;
  document.getElementById('rThreshold').textContent = Number(r.threshold).toFixed(4);
  document.getElementById('rModel').textContent = a.model;
  document.getElementById('rFeatures').textContent = (r.features_used || []).join('  ·  ');

  // animate the gauge
  const g = document.getElementById('gaugeValue');
  const circumference = 2 * Math.PI * 86;
  g.style.strokeDasharray = circumference;
  g.style.strokeDashoffset = circumference;
  g.style.stroke = colour;
  requestAnimationFrame(() => {
    setTimeout(() => { g.style.strokeDashoffset = circumference * (1 - r.risk); }, 90);
  });

  document.getElementById('riskNumber').textContent = pct;
}

/* ---------------- chrome ---------------- */
const nav = document.querySelector('.nav');
const rail = document.getElementById('progressRail');
const railFill = rail.querySelector('span');

function onScroll() {
  nav.classList.toggle('nav--sticky', window.scrollY > 12);
  const max = document.body.scrollHeight - window.innerHeight;
  railFill.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + '%';
}
window.addEventListener('scroll', onScroll, { passive: true });

/* ---------------- boot ---------------- */
document.addEventListener('DOMContentLoaded', () => {
  renderLanding();
  onScroll();

  const hash = location.hash.replace('#', '');
  if (hash && document.getElementById(hash)) {
    if (ASSESSMENTS[hash]) {
      startAssessment(hash, false);
    } else if (hash === 'result') {
      go('result', false);
    } else if (hash === 'assessment') {
      startAssessment(currentType, false);
    } else if (hash === 'landing') {
      // already on landing
    } else {
      // section anchor (method, results…)
      const target = document.getElementById(hash);
      if (target) setTimeout(() => target.scrollIntoView({ behavior: 'smooth' }), 80);
    }
  } else {
    // Push an initial history entry for the landing page so popstate fires correctly
    history.replaceState({ screen: 'landing' }, '', location.href);
  }
});
