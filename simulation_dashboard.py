"""
Live Simulation Dashboard (web) for the RL-based routing project.

Runs entirely on one laptop — no hardware required. Two panels:

  1. RESULTS — the validated evaluation results from results/evaluation_results.json
     (300 episodes per setting, produced by evaluate.py), shown as packet-delivery
     and delay bars with a failure-rate selector (0 / 20 / 40 / 60%). These are the
     project's real, measured numbers: GNN-DQN is the only method that holds 100%
     delivery across every failure rate *and* the lowest end-to-end delay.

  2. LIVE TOPOLOGY — an illustrative real-time animation of one GNN-DQN packet
     hopping across the 10-node network while links fail, recover and change
     congestion (green → red). This is the intuition behind the numbers: a learned
     policy that keeps finding a way through a network that won't sit still. It is
     a qualitative animation; the quantitative comparison lives in panel 1.

Design note (honesty): on a *static* snapshot Dijkstra is delay-optimal by
construction, and this densely-connected topology lets the shortest-path baselines
keep high delivery even under churn — so we do NOT stage a misleading "baselines
collapse" live recomputation. The trustworthy comparison is the validated batch in
panel 1; the animation is explicitly qualitative.

Self-contained: HTML/CSS/JS is inlined and uses only inline SVG, so it works with
no internet connection on the presentation machine.

Usage:
    .venv/bin/python simulation_dashboard.py            # open http://127.0.0.1:5000
    .venv/bin/python simulation_dashboard.py --port 8080 --host 0.0.0.0
"""

import argparse, json, os, sys
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request

from network_rl.env.network_env  import NetworkRoutingEnv, NUM_NODES
from network_rl.agents.gnn_agent import GNNAgent

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "results", "evaluation_results.json")
BAR_ORDER = ["GNN-DQN", "Rainbow", "DQN", "Dijkstra", "ECMP", "Q-Routing", "Random"]

app = Flask(__name__)


class Simulation:
    def __init__(self):
        # Validated evaluation results (real measured numbers).
        with open(RESULTS_JSON) as f:
            self.results = json.load(f)
        self.failure_rates = self.results["failure_rates"]   # e.g. [0,20,40,60]

        # Live illustrative animation: a GNN packet on a churning network.
        self.live_env = NetworkRoutingEnv(use_mm1=True, failure_prob=0.04)
        self.live_env.reset()
        gnn_path = os.path.join(MODELS_DIR, "gnn_trained.pth")
        if not os.path.exists(gnn_path):
            gnn_path = os.path.join(MODELS_DIR, "gnn_seed0.pth")
        self.gnn = GNNAgent(graph=self.live_env.G, edge_list=self.live_env.edge_list)
        self.gnn.load(gnn_path)
        self.gnn.epsilon = 0.0
        self._reset_live_episode()

        pos = nx.kamada_kawai_layout(self.live_env.G)
        xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        self.pos = {int(n): {"x": (p[0]-minx)/(maxx-minx+1e-9),
                             "y": (p[1]-miny)/(maxy-miny+1e-9)}
                    for n, p in pos.items()}
        self.delivered_count = 0
        self.episode_count = 0

    # ── results panel ───────────────────────────────────────────────────────────
    def leaderboard(self, fr_index):
        rows = []
        for name in BAR_ORDER:
            r = self.results["results"].get(name)
            if not r:
                continue
            rows.append({"name": name,
                         "pdr":   round(r["pdr"][fr_index], 3),
                         "delay": round(r["delay"][fr_index], 1)})
        return rows

    # ── live animation ────────────────────────────────────────────────────────────
    def _reset_live_episode(self):
        self.live_env.reset()
        self.live_src = self.live_env._current_node
        self.live_dst = self.live_env._dest_node
        self.live_path = [self.live_src]
        self.live_done = False
        self.live_delivered = False
        self.live_hold = 0

    def step_live(self):
        if self.live_done:
            self.live_hold -= 1
            if self.live_hold <= 0:
                self._reset_live_episode()
            return
        cur = self.live_env._current_node
        action, _ = self.gnn.select_action(self.live_env, cur, self.live_dst)
        _, _, term, trunc, info = self.live_env.step(action)
        self.live_path = info.get("path", self.live_path)
        if term or trunc:
            self.live_done = True
            self.live_delivered = (self.live_env._current_node == self.live_dst)
            self.episode_count += 1
            if self.live_delivered:
                self.delivered_count += 1
            self.live_hold = 3

    def graph_state(self):
        edges = [{"u": int(u), "v": int(v),
                  "congestion": round(float(self.live_env.G.edges[u, v]["congestion"]), 2),
                  "active": bool(self.live_env.G.edges[u, v]["active"])}
                 for (u, v) in self.live_env.edge_list]
        nodes = [{"id": int(n), "x": self.pos[int(n)]["x"], "y": self.pos[int(n)]["y"]}
                 for n in self.live_env.G.nodes()]
        live_pdr = (self.delivered_count / self.episode_count) if self.episode_count else 0.0
        return {
            "nodes": nodes, "edges": edges,
            "live": {"src": self.live_src, "dst": self.live_dst,
                     "path": self.live_path, "current": self.live_env._current_node,
                     "done": self.live_done, "delivered": self.live_delivered,
                     "episodes": self.episode_count, "live_pdr": round(live_pdr, 3)},
        }

    def full_state(self, fr_index=0):
        s = self.graph_state()
        s["failure_rates"] = self.failure_rates
        s["fr_index"] = fr_index
        s["leaderboard"] = self.leaderboard(fr_index)
        s["num_nodes"] = NUM_NODES
        return s


sim = Simulation()


@app.route("/api/state")
def api_state():
    fr = int(request.args.get("fr_index", 0))
    return jsonify(sim.full_state(fr))


@app.route("/api/tick", methods=["POST"])
def api_tick():
    sim.step_live()
    fr = int(request.args.get("fr_index", 0))
    return jsonify(sim.full_state(fr))


@app.route("/")
def index():
    return INDEX_HTML


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>RL Routing — Live Simulation</title>
<style>
  :root{--bg:#0f1420;--panel:#19202e;--ink:#e8edf5;--muted:#8b97ad;--down:#3a4252;--accent:#37d67a;}
  *{box-sizing:border-box;font-family:"Segoe UI",system-ui,sans-serif;}
  body{margin:0;background:var(--bg);color:var(--ink);}
  header{padding:14px 22px;border-bottom:1px solid #232b3b;}
  h1{font-size:19px;margin:0;font-weight:600;}
  .sub{color:var(--muted);font-size:13px;}
  .wrap{display:flex;gap:16px;padding:16px 22px;flex-wrap:wrap;}
  .card{background:var(--panel);border:1px solid #232b3b;border-radius:12px;padding:16px;}
  #left{flex:1;min-width:400px;}
  #right{flex:1;min-width:440px;}
  svg{width:100%;height:500px;display:block;}
  .ctl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;}
  button{background:#212a3b;color:var(--ink);border:1px solid #313c52;border-radius:8px;padding:7px 12px;font-size:14px;cursor:pointer;}
  button:hover{background:#2b3650;} button.on{background:#1f6feb;border-color:#1f6feb;}
  .row{display:flex;align-items:center;gap:10px;margin:9px 0;}
  .name{width:92px;font-size:14px;font-weight:600;}
  .name.win{color:var(--accent);}
  .barbg{flex:1;background:#121824;border:1px solid #2a3346;border-radius:6px;height:24px;position:relative;overflow:hidden;}
  .bar{height:100%;border-radius:5px 0 0 5px;transition:width .5s ease, background .5s;}
  .barlabel{position:absolute;right:8px;top:3px;font-size:12px;color:#fff;}
  .delay{width:74px;text-align:right;font-size:13px;color:var(--muted);}
  .node{fill:#2a3650;stroke:#5b6b8c;stroke-width:2;}
  .node.src{fill:#1f6feb;stroke:#7fb1ff;} .node.dst{fill:#8957e5;stroke:#c4a7ff;}
  .node.cur{fill:var(--accent);stroke:#bff5d2;}
  .nodelabel{fill:#fff;font-size:13px;font-weight:700;text-anchor:middle;pointer-events:none;}
  .legend{font-size:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;}
  .tag{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:middle;}
  .hint{color:var(--muted);font-size:12px;margin-top:8px;}
  .pill{font-size:12px;padding:3px 9px;border-radius:999px;background:#212a3b;border:1px solid #313c52;}
</style></head>
<body>
<header>
  <h1>RL-Based Adaptive Routing — Live Simulation</h1>
  <span class="sub">Validated evaluation (300 episodes/setting) + live GNN-DQN packet animation on a dynamic 10-node network</span>
</header>

<div class="wrap">
  <div class="card" id="left">
    <div class="ctl">
      <strong>Link-failure rate</strong>
      <span id="frbtns"></span>
    </div>
    <div class="sub" style="margin-bottom:6px">Packet Delivery Ratio &nbsp;·&nbsp; <code>results/evaluation_results.json</code></div>
    <div id="bars"></div>
    <p class="hint"><span style="color:var(--accent)">GNN-DQN</span> is the only
    method that keeps <b>100% delivery at every failure rate</b> while also achieving
    the <b>lowest end-to-end delay</b> (right column, ms). These are the real measured
    results — switch the failure rate to see they hold across conditions.</p>
  </div>

  <div class="card" id="right">
    <div class="sub" id="livecap">Live GNN-DQN packet routing (illustrative)</div>
    <svg id="svg" viewBox="0 0 1000 500" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="legend">
      <span><span class="tag" style="background:#1f6feb"></span>source</span>
      <span><span class="tag" style="background:#8957e5"></span>destination</span>
      <span><span class="tag" style="background:var(--accent)"></span>packet / path</span>
      <span><span class="tag" style="background:var(--down)"></span>failed link</span>
      <span>link colour = congestion (green → red)</span>
    </div>
    <p class="hint" id="livepdr"></p>
  </div>
</div>

<script>
const W=1000,H=500,R=20,PAD=60;
let state=null, frIndex=0, playing=true;

function px(n){return PAD+n.x*(W-2*PAD);}
function py(n){return PAD+n.y*(H-2*PAD);}
function congColor(c,active){
  if(!active) return getComputedStyle(document.documentElement).getPropertyValue('--down');
  const r=Math.round(55+200*c),g=Math.round(200-150*c),b=90;return `rgb(${r},${g},${b})`;
}
function edgeKey(u,v){return u<v?`${u}-${v}`:`${v}-${u}`;}
function pathEdges(p){const s=new Set();for(let i=0;i<p.length-1;i++)s.add(edgeKey(p[i],p[i+1]));return s;}

function drawGraph(){
  const svg=document.getElementById('svg'); svg.innerHTML='';
  const byId={}; state.nodes.forEach(n=>byId[n.id]=n);
  const live=state.live; const onPath=pathEdges(live.path);
  state.edges.forEach(e=>{
    const a=byId[e.u],b=byId[e.v];
    const l=document.createElementNS('http://www.w3.org/2000/svg','line');
    l.setAttribute('x1',px(a));l.setAttribute('y1',py(a));
    l.setAttribute('x2',px(b));l.setAttribute('y2',py(b));
    l.setAttribute('stroke',congColor(e.congestion,e.active));
    l.setAttribute('stroke-width',e.active?5:3);l.setAttribute('stroke-linecap','round');
    if(!e.active)l.setAttribute('stroke-dasharray','6 7');
    svg.appendChild(l);
  });
  const acc=getComputedStyle(document.documentElement).getPropertyValue('--accent');
  state.edges.forEach(e=>{
    if(!onPath.has(edgeKey(e.u,e.v)))return;
    const a=byId[e.u],b=byId[e.v];
    const l=document.createElementNS('http://www.w3.org/2000/svg','line');
    l.setAttribute('x1',px(a));l.setAttribute('y1',py(a));
    l.setAttribute('x2',px(b));l.setAttribute('y2',py(b));
    l.setAttribute('stroke',acc);l.setAttribute('stroke-width',7);
    l.setAttribute('stroke-linecap','round');l.setAttribute('opacity',0.9);
    svg.appendChild(l);
  });
  state.nodes.forEach(n=>{
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',px(n));c.setAttribute('cy',py(n));c.setAttribute('r',R);
    let cls='node'; if(n.id===live.src)cls+=' src'; if(n.id===live.dst)cls+=' dst';
    if(n.id===live.current && !live.done)cls+=' cur';
    c.setAttribute('class',cls); svg.appendChild(c);
    const t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',px(n));t.setAttribute('y',py(n)+5);t.setAttribute('class','nodelabel');
    t.textContent=n.id; svg.appendChild(t);
  });
  let cap=`Live GNN-DQN packet: ${live.src} → ${live.dst} &nbsp; path: ${live.path.join(' → ')}`;
  if(live.done) cap+= live.delivered?' &nbsp; ✅ delivered':' &nbsp; ⛔ not delivered';
  document.getElementById('livecap').innerHTML=cap;
  document.getElementById('livepdr').textContent=
    `Live delivery so far: ${(live.live_pdr*100).toFixed(0)}% over ${live.episodes} episodes (illustrative — quantitative results are on the left).`;
}

function drawBars(){
  const box=document.getElementById('bars'); box.innerHTML='';
  state.leaderboard.forEach(r=>{
    const win=r.name==='GNN-DQN';
    const pct=Math.round(r.pdr*100); const hue=120*r.pdr;
    const row=document.createElement('div'); row.className='row';
    row.innerHTML=
      `<div class="name ${win?'win':''}">${r.name}</div>`+
      `<div class="barbg"><div class="bar" style="width:${pct}%;background:hsl(${hue},65%,45%)"></div>`+
      `<span class="barlabel">${pct}%</span></div>`+
      `<div class="delay">${r.delay?r.delay.toFixed(0)+' ms':'—'}</div>`;
    box.appendChild(row);
  });
}

function drawFrButtons(){
  const box=document.getElementById('frbtns'); box.innerHTML='';
  state.failure_rates.forEach((fr,i)=>{
    const b=document.createElement('button');
    b.textContent=fr+'%'; if(i===frIndex)b.className='on';
    b.onclick=()=>{frIndex=i; refresh();};
    box.appendChild(b);
  });
}

function render(){ if(!state)return; drawFrButtons(); drawBars(); drawGraph(); }

async function refresh(){
  const r=await fetch('/api/state?fr_index='+frIndex); state=await r.json(); render();
}
async function tick(){
  if(!playing)return;
  const r=await fetch('/api/tick?fr_index='+frIndex,{method:'POST'}); state=await r.json(); render();
}

(async function(){ await refresh(); setInterval(tick, 750); })();
</script>
</body></html>
"""


def main():
    p = argparse.ArgumentParser(description="Live web dashboard for the RL routing simulation")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    shown = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Dashboard ready → http://{shown}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
