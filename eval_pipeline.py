#!/usr/bin/env python3
"""
Local dry-run of the agent/eval dashboard.

This runs the *real* evaluation-harness logic, but against MOCK model
endpoints instead of live Nebius vLLM endpoints. Everything downstream of the
model call -- scoring, latency capture, cost derivation, the results artifact,
and the dashboard -- is exactly what you'd ship.

To go live, you replace `MockEndpoint.chat()` with a real OpenAI-compatible
call against your Nebius endpoint URL. That is the only thing that changes:

    from openai import OpenAI
    client = OpenAI(base_url=f"http://{endpoint_ip}/v1", api_key=AUTH_TOKEN)
    resp = client.chat.completions.create(model=model_id, messages=msgs)

Outputs: results.json (the artifact) and dashboard.html (reads the artifact).
"""

import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).parent

# --- The public "benchmark": small, deterministic, exact-match scorable -------
TASKS = [
    {"q": "What is 13 + 29?",                         "a": "42",   "wrong": "41"},
    {"q": "What is 7 * 8?",                            "a": "56",   "wrong": "54"},
    {"q": "What is the capital of France?",           "a": "Paris","wrong": "Lyon"},
    {"q": "How many sides does a hexagon have?",       "a": "6",    "wrong": "8"},
    {"q": "What is 144 / 12?",                         "a": "12",   "wrong": "14"},
    {"q": "Extract the year: 'Released in 1994.'",     "a": "1994", "wrong": "1949"},
    {"q": "What color is a clear daytime sky?",        "a": "blue", "wrong": "grey"},
    {"q": "What is 100 - 37?",                         "a": "63",   "wrong": "73"},
    {"q": "What is the plural of 'mouse'?",            "a": "mice", "wrong": "mouses"},
    {"q": "What is 2 to the power of 5?",              "a": "32",   "wrong": "25"},
    {"q": "Chemical symbol for gold?",                 "a": "Au",   "wrong": "Go"},
    {"q": "What is 15% of 200?",                       "a": "30",   "wrong": "15"},
]

# --- Mock model endpoints (stand-ins for Nebius vLLM endpoints) ---------------
# skill   = probability of answering correctly (bigger model => higher)
# lat_*   = response latency profile in seconds (bigger model => slower)
# rate_hr = endpoint $/hour for its preset (used to derive cost)
MODELS = [
    {"id": "qwen2.5-0.5b-instruct", "skill": 0.55, "lat_mean": 0.22, "lat_sd": 0.05,
     "rate_hr": 1.55, "preset": "gpu-l40s-a / 1gpu-8vcpu-32gb"},
    {"id": "qwen2.5-1.5b-instruct", "skill": 0.78, "lat_mean": 0.46, "lat_sd": 0.08,
     "rate_hr": 1.55, "preset": "gpu-l40s-a / 1gpu-8vcpu-32gb"},
    {"id": "llama-3.2-3b-instruct", "skill": 0.95, "lat_mean": 1.05, "lat_sd": 0.16,
     "rate_hr": 1.55, "preset": "gpu-l40s-a / 1gpu-8vcpu-32gb"},
]


class MockEndpoint:
    """Deterministic stand-in for a vLLM OpenAI-compatible endpoint."""

    def __init__(self, spec):
        self.spec = spec

    def chat(self, task, seed):
        rng = random.Random(seed)
        correct = rng.random() < self.spec["skill"]
        answer = task["a"] if correct else task["wrong"]
        latency = max(0.03, rng.gauss(self.spec["lat_mean"], self.spec["lat_sd"]))
        out_tokens = rng.randint(6, 24)          # pretend output length
        return {"answer": answer, "latency_s": latency, "out_tokens": out_tokens}


def normalize(s):
    return s.strip().lower().rstrip(".")


def run_eval():
    samples = []
    per_model = {m["id"]: {"lat": [], "cost_per_1k": [], "req_cost": [], "correct": 0}
                 for m in MODELS}
    endpoints = {m["id"]: MockEndpoint(m) for m in MODELS}
    rate = {m["id"]: m["rate_hr"] for m in MODELS}

    for i, task in enumerate(TASKS):
        row = {"q": task["q"], "expected": task["a"], "answers": {}}
        for m in MODELS:
            mid = m["id"]
            out = endpoints[mid].chat(task, seed=f"{mid}:{i}")
            ok = normalize(out["answer"]) == normalize(task["a"])
            # cost derivation: per-second billing * seconds the GPU was busy
            req_cost = rate[mid] / 3600.0 * out["latency_s"]
            cost_per_1k = req_cost / out["out_tokens"] * 1000.0
            per_model[mid]["lat"].append(out["latency_s"])
            per_model[mid]["cost_per_1k"].append(cost_per_1k)
            per_model[mid]["req_cost"].append(req_cost)
            per_model[mid]["correct"] += int(ok)
            row["answers"][mid] = {"text": out["answer"], "correct": ok,
                                   "latency_s": round(out["latency_s"], 3)}
        samples.append(row)

    leaderboard = []
    for m in MODELS:
        mid = m["id"]
        d = per_model[mid]
        n = len(TASKS)
        lat_sorted = sorted(d["lat"])
        p95 = lat_sorted[min(n - 1, int(round(0.95 * (n - 1))))]
        leaderboard.append({
            "model": mid,
            "preset": m["preset"],
            "accuracy": round(d["correct"] / n, 4),
            "correct": d["correct"],
            "n": n,
            "mean_latency_s": round(statistics.mean(d["lat"]), 3),
            "p95_latency_s": round(p95, 3),
            "cost_per_1k_tokens_usd": round(statistics.mean(d["cost_per_1k"]), 5),
            "total_run_cost_usd": round(sum(d["req_cost"]), 5),
        })

    leaderboard.sort(key=lambda r: r["accuracy"], reverse=True)
    return {
        "meta": {"mode": "local-dry-run", "n_models": len(MODELS),
                 "n_tasks": len(TASKS), "benchmark": "mini-qa (exact match)"},
        "leaderboard": leaderboard,
        "samples": samples,
    }


def render_dashboard(results, out_path):
    data_json = json.dumps(results)
    html = DASHBOARD_TEMPLATE.replace("__DATA__", data_json)
    out_path.write_text(html, encoding="utf-8")


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model eval dashboard</title>
<style>
  :root{
    --bg:#0E1217; --panel:#161C24; --panel2:#1C242E; --line:#28323E;
    --txt:#E6EBF0; --dim:#8A97A6; --faint:#5A6675;
    --cyan:#4DD2C4; --amber:#E0A458; --violet:#8B9BF0;
    --good:#6BD08B; --bad:#E0728A;
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace;
    --sans:ui-sans-serif,system-ui,"Inter",-apple-system,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);font-family:var(--sans);
       line-height:1.5;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1040px;margin:0 auto;padding:32px 22px 64px}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.22em;
           text-transform:uppercase;color:var(--cyan);margin:0 0 8px}
  h1{font-size:30px;font-weight:650;margin:0 0 6px;letter-spacing:-.01em}
  .meta{font-family:var(--mono);font-size:12.5px;color:var(--dim);margin:0}
  .meta b{color:var(--txt);font-weight:500}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:26px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
         padding:18px 18px 20px;animation:rise .4s ease both}
  .panel.full{grid-column:1 / -1}
  .plabel{font-family:var(--mono);font-size:10.5px;letter-spacing:.18em;
          text-transform:uppercase;color:var(--faint);margin:0 0 14px}
  table{width:100%;border-collapse:collapse;font-size:13.5px}
  th{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
     color:var(--dim);text-align:right;font-weight:500;padding:0 0 9px;border-bottom:1px solid var(--line)}
  th.l,td.l{text-align:left}
  td{padding:10px 0;border-bottom:1px solid var(--panel2);font-family:var(--mono)}
  tr:last-child td{border-bottom:none}
  .model-name{color:var(--txt);font-weight:500}
  .preset{color:var(--faint);font-size:11px}
  .best{color:var(--cyan)}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:middle}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .ax{stroke:var(--line);stroke-width:1}
  .grid-l{stroke:var(--panel2);stroke-width:1}
  .axlab{fill:var(--dim);font-family:var(--mono);font-size:10px}
  .axtitle{fill:var(--faint);font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase}
  .ptlab{fill:var(--txt);font-family:var(--mono);font-size:11px}
  .frontier{stroke:var(--cyan);stroke-width:1.4;stroke-dasharray:4 4;fill:none;opacity:.7}
  .bar{fill:var(--violet)}
  .barbg{fill:var(--panel2)}
  .barval{fill:var(--txt);font-family:var(--mono);font-size:11px}
  .barname{fill:var(--dim);font-family:var(--mono);font-size:11px}
  select{font-family:var(--mono);font-size:13px;background:var(--panel2);color:var(--txt);
         border:1px solid var(--line);border-radius:8px;padding:8px 10px;width:100%;margin-bottom:14px}
  .insp-q{font-size:14px;color:var(--txt);margin:2px 0 14px;padding:10px 12px;
          background:var(--panel2);border-radius:8px;border:1px solid var(--line)}
  .insp-q .exp{font-family:var(--mono);color:var(--cyan);font-size:12px}
  .ans{display:flex;align-items:center;justify-content:space-between;gap:12px;
       padding:9px 0;border-bottom:1px solid var(--panel2);font-family:var(--mono);font-size:13px}
  .ans:last-child{border-bottom:none}
  .ans .m{color:var(--dim)}
  .ans .v{display:flex;align-items:center;gap:9px}
  .mark{font-weight:700}
  .ok{color:var(--good)} .no{color:var(--bad)}
  .note{font-family:var(--mono);font-size:11px;color:var(--faint);margin:22px 0 0;
        padding-top:16px;border-top:1px solid var(--line)}
  @keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  @media (prefers-reduced-motion:reduce){.panel{animation:none}}
  @media (max-width:720px){.grid{grid-template-columns:1fr}h1{font-size:24px}}
  :focus-visible{outline:2px solid var(--cyan);outline-offset:2px}
</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">Local dry-run · mock endpoints</p>
  <h1>Model eval dashboard</h1>
  <p class="meta" id="meta"></p>

  <div class="grid">
    <div class="panel full" style="animation-delay:.02s">
      <p class="plabel">Leaderboard</p>
      <table id="board"></table>
    </div>

    <div class="panel" style="animation-delay:.06s">
      <p class="plabel">Quality vs cost — what's worth it</p>
      <div id="scatter"></div>
    </div>

    <div class="panel" style="animation-delay:.08s">
      <p class="plabel">Mean latency</p>
      <div id="latency"></div>
    </div>

    <div class="panel full" style="animation-delay:.1s">
      <p class="plabel">Sample inspector</p>
      <select id="picker" aria-label="Choose a task"></select>
      <div class="insp-q" id="iq"></div>
      <div id="ians"></div>
    </div>
  </div>

  <p class="note">Mock data from a local dry-run. To go live, point the client at your
  Nebius vLLM endpoint URLs — the scoring, cost math, and this view stay identical.</p>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const COLORS = ["#4DD2C4","#8B9BF0","#E0A458"];
const colorFor = (i)=>COLORS[i % COLORS.length];

// meta
document.getElementById("meta").innerHTML =
  `<b>${D.meta.n_models}</b> endpoints · <b>${D.meta.n_tasks}</b> tasks · `+
  `benchmark <b>${D.meta.benchmark}</b> · mode <b>${D.meta.mode}</b>`;

// leaderboard
const lb = D.leaderboard;
const bestAcc = Math.max(...lb.map(r=>r.accuracy));
const bestLat = Math.min(...lb.map(r=>r.mean_latency_s));
const bestCost = Math.min(...lb.map(r=>r.cost_per_1k_tokens_usd));
const modelIndex = {};
lb.forEach((r,i)=>modelIndex[r.model]=i);
const board = document.getElementById("board");
board.innerHTML =
  `<tr><th class="l">Model</th><th>Accuracy</th><th>Mean lat</th>
   <th>p95 lat</th><th>$ / 1k tok</th><th>Run cost</th></tr>` +
  lb.map((r)=>{
    const ci = modelIndex[r.model];
    return `<tr>
      <td class="l"><span class="dot" style="background:${colorFor(ci)}"></span>
        <span class="model-name">${r.model}</span><br>
        <span class="preset" style="margin-left:16px">${r.preset}</span></td>
      <td class="${r.accuracy===bestAcc?'best':''}">${(r.accuracy*100).toFixed(0)}%</td>
      <td class="${r.mean_latency_s===bestLat?'best':''}">${r.mean_latency_s.toFixed(2)}s</td>
      <td>${r.p95_latency_s.toFixed(2)}s</td>
      <td class="${r.cost_per_1k_tokens_usd===bestCost?'best':''}">$${r.cost_per_1k_tokens_usd.toFixed(4)}</td>
      <td>$${r.total_run_cost_usd.toFixed(4)}</td>
    </tr>`;
  }).join("");

// scatter: x = cost/1k, y = accuracy
(function(){
  const W=520,H=320,m={t:18,r:60,b:46,l:46};
  const pts = lb.map(r=>({x:r.cost_per_1k_tokens_usd, y:r.accuracy,
                          label:r.model.replace("-instruct",""), i:modelIndex[r.model]}));
  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  const xmin=Math.min(...xs)*0.85, xmax=Math.max(...xs)*1.1;
  const ymin=Math.max(0,Math.min(...ys)-0.15), ymax=1.0;
  const X=v=>m.l+(v-xmin)/(xmax-xmin)*(W-m.l-m.r);
  const Y=v=>H-m.b-(v-ymin)/(ymax-ymin)*(H-m.t-m.b);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Quality versus cost scatter">`;
  // gridlines + y ticks
  for(let k=0;k<=4;k++){
    const yv=ymin+(ymax-ymin)*k/4, y=Y(yv);
    s+=`<line class="grid-l" x1="${m.l}" y1="${y}" x2="${W-m.r}" y2="${y}"/>`;
    s+=`<text class="axlab" x="${m.l-8}" y="${y+3}" text-anchor="end">${(yv*100).toFixed(0)}%</text>`;
  }
  // x ticks
  for(let k=0;k<=3;k++){
    const xv=xmin+(xmax-xmin)*k/3, x=X(xv);
    s+=`<text class="axlab" x="${x}" y="${H-m.b+16}" text-anchor="middle">$${xv.toFixed(4)}</text>`;
  }
  s+=`<line class="ax" x1="${m.l}" y1="${H-m.b}" x2="${W-m.r}" y2="${H-m.b}"/>`;
  s+=`<line class="ax" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H-m.b}"/>`;
  s+=`<text class="axtitle" x="${(m.l+W-m.r)/2}" y="${H-6}" text-anchor="middle">cost / 1k tokens</text>`;
  s+=`<text class="axtitle" transform="translate(13,${(m.t+H-m.b)/2}) rotate(-90)" text-anchor="middle">accuracy</text>`;
  // frontier: non-dominated points (low cost & high accuracy), sorted by cost
  const sorted=[...pts].sort((a,b)=>a.x-b.x);
  const front=[]; let bestY=-1;
  for(const p of sorted){ if(p.y>bestY){front.push(p); bestY=p.y;} }
  if(front.length>1){
    s+=`<polyline class="frontier" points="${front.map(p=>`${X(p.x)},${Y(p.y)}`).join(" ")}"/>`;
  }
  // points
  for(const p of pts){
    s+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="6.5" fill="${colorFor(p.i)}" stroke="#0E1217" stroke-width="1.5"/>`;
    const anchor = X(p.x)>W-m.r-90 ? "end" : "start";
    const dx = anchor==="end" ? -11 : 11;
    s+=`<text class="ptlab" x="${X(p.x)+dx}" y="${Y(p.y)+4}" text-anchor="${anchor}">${p.label}</text>`;
  }
  s+="</svg>";
  document.getElementById("scatter").innerHTML=s;
})();

// latency bars
(function(){
  const rows=[...lb].sort((a,b)=>a.mean_latency_s-b.mean_latency_s);
  const W=460,rowH=46,H=rows.length*rowH+10,m={l:8,r:8};
  const max=Math.max(...rows.map(r=>r.mean_latency_s))*1.15;
  const barW=W-150;
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mean latency bars">`;
  rows.forEach((r,i)=>{
    const y=i*rowH+8, ci=modelIndex[r.model];
    const w=r.mean_latency_s/max*barW;
    s+=`<text class="barname" x="0" y="${y+13}">${r.model.replace("-instruct","")}</text>`;
    s+=`<rect class="barbg" x="0" y="${y+20}" width="${barW}" height="10" rx="5"/>`;
    s+=`<rect x="0" y="${y+20}" width="${w}" height="10" rx="5" fill="${colorFor(ci)}"/>`;
    s+=`<text class="barval" x="${barW+10}" y="${y+29}">${r.mean_latency_s.toFixed(2)}s</text>`;
  });
  s+="</svg>";
  document.getElementById("latency").innerHTML=s;
})();

// sample inspector
(function(){
  const picker=document.getElementById("picker");
  D.samples.forEach((s,i)=>{
    const o=document.createElement("option");
    o.value=i; o.textContent=`${i+1}. ${s.q}`;
    picker.appendChild(o);
  });
  function show(idx){
    const s=D.samples[idx];
    document.getElementById("iq").innerHTML =
      `${s.q} &nbsp;<span class="exp">expected: ${s.expected}</span>`;
    document.getElementById("ians").innerHTML = lb.map(r=>{
      const a=s.answers[r.model], ci=modelIndex[r.model];
      const mk=a.correct?'<span class="mark ok">✓</span>':'<span class="mark no">✗</span>';
      return `<div class="ans">
        <span class="m"><span class="dot" style="background:${colorFor(ci)}"></span>${r.model}</span>
        <span class="v">${mk}<span style="color:${a.correct?'var(--good)':'var(--bad)'}">${a.text}</span>
          <span style="color:var(--faint)">${a.latency_s.toFixed(2)}s</span></span>
      </div>`;
    }).join("");
  }
  picker.addEventListener("change",e=>show(+e.target.value));
  show(0);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    results = run_eval()
    (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    render_dashboard(results, HERE / "dashboard.html")

    print("=== EVAL COMPLETE (local dry-run) ===")
    print(f"benchmark: {results['meta']['benchmark']}  "
          f"models: {results['meta']['n_models']}  tasks: {results['meta']['n_tasks']}\n")
    hdr = f"{'model':<26}{'acc':>6}{'mean_lat':>10}{'$/1k_tok':>11}{'run_cost':>11}"
    print(hdr); print("-" * len(hdr))
    for r in results["leaderboard"]:
        print(f"{r['model']:<26}{r['accuracy']*100:>5.0f}%"
              f"{r['mean_latency_s']:>9.2f}s"
              f"{'$'+format(r['cost_per_1k_tokens_usd'],'.4f'):>11}"
              f"{'$'+format(r['total_run_cost_usd'],'.4f'):>11}")
    print("\nwrote: results.json, dashboard.html")
