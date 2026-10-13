'use strict';
const $ = (id) => document.getElementById(id);
const indexURL = 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/';
const sample = [
  'treatment,age,bmi,outcome',
  '1,62,27,11','0,61,26,9','1,45,22,15','0,46,23,13',
  '1,55,24,16','0,56,25,14','1,51,28,12','0,52,27,10',
  '1,68,29,17','0,67,30,15','1,39,21,20','0,40,22,18'
].join('\n');
let runtime = null;
let resultForExport = null;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function parseCsv(text) {
  const records = [];
  let row = [], field = '', quoted = false;
  const input = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (ch === '"') {
      if (quoted && input[i + 1] === '"') { field += '"'; i++; }
      else if (quoted) quoted = false;
      else if (!field) quoted = true;
      else throw Error('Unexpected quote in CSV field.');
    } else if (ch === ',' && !quoted) {
      row.push(field); field = '';
    } else if ((ch === '\n' || ch === '\r') && !quoted) {
      if (ch === '\r' && input[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.some(v => v.trim())) records.push(row);
      row = [];
    } else field += ch;
  }
  if (quoted) throw Error('Unterminated quoted CSV value.');
  row.push(field);
  if (row.some(v => v.trim())) records.push(row);
  if (records.length < 3) throw Error('CSV needs a header and at least two subjects.');
  const headers = records.shift().map(v => v.trim());
  if (headers.some(v => !v) || new Set(headers.map(v => v.toLowerCase())).size !== headers.length) {
    throw Error('CSV column names must be nonempty and unique.');
  }
  if (records.some(row => row.length !== headers.length)) throw Error('CSV rows have inconsistent column counts.');
  return {headers, records};
}
function dataFromCsv(text) {
  const {headers, records} = parseCsv(text);
  const lower = headers.map(h => h.toLowerCase());
  const treatmentCol = ['treatment','treated','group','arm','exposure'].map(x => lower.indexOf(x)).find(x => x !== -1);
  if (treatmentCol === undefined) throw Error('A treatment column (0 or 1) is required.');
  const outcomeCol = ['outcome','y','result','response'].map(x => lower.indexOf(x)).find(x => x !== -1);
  const covIds = headers.map((_, i) => i).filter(i => i !== treatmentCol && i !== outcomeCol);
  function numeric(row, col, index) {
    const raw = row[col].trim();
    if (!raw) throw Error('Missing value: row ' + (index + 2) + ', column ' + headers[col]);
    const n = Number(raw);
    if (!Number.isFinite(n)) throw Error('Invalid number: row ' + (index + 2) + ', column ' + headers[col]);
    return n;
  }
  const treatment = records.map((r,i) => numeric(r,treatmentCol,i));
  if (treatment.some(v => v !== 0 && v !== 1)) throw Error('Treatment must contain only 0 and 1.');
  if (!treatment.includes(0) || !treatment.includes(1)) throw Error('Both treatment arms are required.');
  return {
    treatment,
    covariates:records.map((r,i) => covIds.map(col => numeric(r,col,i))),
    outcome:outcomeCol === undefined ? null : records.map((r,i) => numeric(r,outcomeCol,i)),
    labels:covIds.map(i => headers[i]),
    records,headers
  };
}
function fmt(value) {
  return value === null || value === undefined ? 'N/A' : Number(value).toFixed(4);
}
function rowsToTable(id, rows) {
  const body = $(id);
  body.replaceChildren();
  for (const values of rows) {
    const tr = document.createElement('tr');
    for (const v of values) {
      const td = document.createElement('td');
      td.textContent = String(v);
      tr.append(td);
    }
    body.append(tr);
  }
}
async function getRuntime() {
  if (runtime) return runtime;
  status('Loading Python and the matching engine (first use requires internet access)…');
  const { loadPyodide } = await import(indexURL + 'pyodide.mjs');
  const py = await loadPyodide({indexURL});
  const response = await fetch('ps_matcher.py');
  if (!response.ok) throw Error('Cannot load ps_matcher.py (' + response.status + ').');
  py.FS.writeFile('/ps_matcher.py', await response.text());
  await py.runPythonAsync('import ps_matcher');
  runtime = py;
  return py;
}
function render(data, result) {
  $('treated').textContent = result.n_treated;
  $('controls').textContent = result.n_control;
  $('pairs').textContent = (result.matching.matched_pairs || []).length;
  const effect = result.effect ? ' Descriptive ATT: ' + fmt(result.effect.effect) + '.'
    + (result.effect.se === null ? ' Uncertainty not estimated.' : ' Approximate 95% CI: [' + fmt(result.effect.ci_lower) + ', ' + fmt(result.effect.ci_upper) + '].') : '';
  $('summary').textContent = 'Method: ' + result.method + '. Matched treated subjects: '
    + new Set((result.matching.matched_pairs || []).map(p => p[0])).size + '.' + effect
    + (result.ps_model_warning ? ' WARNING: ' + result.ps_model_warning : '');
  rowsToTable('balance',result.balance.balance.map(v => [v.covariate,fmt(v.smd_before),fmt(v.smd_after)]));
  rowsToTable('matches',result.matching.matched_pairs || []);
  resultForExport = {data,result};
  $('download').disabled = false;
}
function escapedCsv(value) {
  const s = String(value === null || value === undefined ? '' : value);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g,'""') + '"' : s;
}
function downloadCsv() {
  if (!resultForExport) return;
  const {data,result} = resultForExport;
  const pairs = result.matching.matched_pairs || [];
  const headers = data.headers.concat(['propensity_score','matched_indices','stratum']);
  const strata = result.matching.strata ? result.matching.strata.strata : null;
  const lines = [headers.map(escapedCsv).join(',')];
  for (let i=0;i<data.records.length;i++) {
    const partners = pairs.filter(p => p[0] === i || p[1] === i)
      .map(p => p[0] === i ? p[1] : p[0]).join(';');
    const record = data.records[i].concat([result.propensity_scores[i],partners,strata ? strata[i] : '']);
    lines.push(record.map(escapedCsv).join(','));
  }
  const url = URL.createObjectURL(new Blob([lines.join('\r\n') + '\r\n'],{type:'text/csv;charset=utf-8'}));
  const anchor = document.createElement('a');
  anchor.href = url;anchor.download = 'propensity_matching_results.csv';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}
async function run() {
  $('run').disabled = true;
  $('download').disabled = true;
  resultForExport = null;
  $('treated').textContent = '—';
  $('controls').textContent = '—';
  $('pairs').textContent = '—';
  $('summary').textContent = 'No current results.';
  rowsToTable('balance', []);
  rowsToTable('matches', []);
  try {
    if ($('csv').value.length > 5 * 1024 * 1024) throw Error('Input exceeds the 5 MB limit.');
    const data = dataFromCsv($('csv').value);
    const caliper = Number($('caliper').value);
    const ratio = Number($('ratio').value);
    const n_strata = Number($('strata').value);
    if ($('caliper').value.trim()==='' || !Number.isFinite(caliper) || caliper < 0) throw Error('Caliper must be non-negative.');
    if (!Number.isSafeInteger(ratio) || ratio < 1) throw Error('Ratio must be a positive integer.');
    if (!Number.isSafeInteger(n_strata) || n_strata < 1) throw Error('Strata must be a positive integer.');
    const py = await getRuntime();
    status('Running propensity score estimation and matching…');
    const payload = {treatment:data.treatment,covariates:data.covariates,outcome:data.outcome,
      labels:data.labels,method:$('method').value,caliper,ratio,n_strata,replace:$('replace').checked};
    py.globals.set('browser_payload',JSON.stringify(payload));
    let output;
    try {
      output = await py.runPythonAsync(
        'import json, math, ps_matcher\n' +
        'd = json.loads(browser_payload)\n' +
        'result = ps_matcher.propensity_score_match(**d)\n' +
        'def safe(obj):\n' +
        '    if isinstance(obj, float) and math.isinf(obj): return "Infinity"\n' +
    '    if isinstance(obj, float) and not math.isfinite(obj): return None\n' +
        '    if isinstance(obj, dict): return {k: safe(v) for k,v in obj.items()}\n' +
        '    if isinstance(obj, (list,tuple)): return [safe(v) for v in obj]\n' +
        '    return obj\n' +
        'json.dumps(safe(result))'
      );
    } finally {py.globals.delete('browser_payload');}
    const result = JSON.parse(output);
    render(data,result);
    status('Analysis complete. Results were calculated locally using ps_matcher.py.');
  } catch (error) {
    status('Analysis failed: ' + (error && error.message ? error.message : String(error)),true);
  } finally {$('run').disabled = false;}
}
$('sample').addEventListener('click',() => {$('csv').value = sample;$('dataset').value = '';});
$('dataset').addEventListener('change',async event => {
  const file = event.target.files && event.target.files[0];
  if (file) {
    if (file.size > 5 * 1024 * 1024) {status('CSV files larger than 5 MB are not supported in this browser interface.',true);return;}
    try {$('csv').value = await file.text();status('CSV loaded locally.');}
    catch(error) {status('Could not read selected CSV file.',true);}
  }
});
$('run').addEventListener('click',run);
$('download').addEventListener('click',downloadCsv);
$('csv').value=sample;
