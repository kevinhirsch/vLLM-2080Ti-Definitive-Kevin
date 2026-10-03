# gateway-part: the two inline HTML pages: DASHBOARD_HTML (legacy fallback, served only when gateway_dashboard.html is missing or unreadable) and ALIASES_HTML (/gateway/aliases/page)
# gateway-part: executed inside keepalive-shim.py's own namespace by _include_gateway_part() -- not an
# gateway-part: importable module. Names here are the shim's globals. See gateway_parts.py.
DASHBOARD_HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>vLLM Gateway</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--dim:#8b949e;--grn:#3fb950;--amb:#d29922;--red:#f85149;--blu:#58a6ff;--acc:#a371f7;--hover:#1c2129}
@media (prefers-color-scheme:light){
 :root{--bg:#f6f8fa;--card:#ffffff;--bd:#d0d7de;--fg:#1f2328;--dim:#57606a;--grn:#1a7f37;--amb:#9a6700;--red:#cf222e;--blu:#0969da;--acc:#8250df;--hover:#eef1f4}
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
.mono,.v,td.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1280px;margin:0 auto;padding:0 16px 40px}
.rz{color:var(--dim)}
a{color:var(--blu);text-decoration:none}a:hover{opacity:.85}
:focus-visible{outline:2px solid var(--blu);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.001ms!important;transition-duration:.001ms!important}}

/* header */
header.top{position:sticky;top:0;z-index:6;background:rgba(13,17,23,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--bd);margin:0 -16px 0;padding:10px 16px;display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px}
@media (prefers-color-scheme:light){header.top{background:rgba(246,248,250,.94)}}
header h1{font-size:15px;margin:0;font-weight:650;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;vertical-align:middle}
.dot.up{background:var(--grn);box-shadow:0 0 6px var(--grn)}
.dot.down{background:var(--red);box-shadow:0 0 6px var(--red)}
.dot.warn{background:var(--amb);box-shadow:0 0 6px var(--amb)}
.stamp{font-size:11px;padding:2px 9px;border-radius:10px;font-weight:600}
.stamp.local{background:rgba(63,185,80,.2);color:var(--grn)}
.stamp.remote{background:rgba(210,153,34,.25);color:var(--amb)}
.stamp.locked{background:rgba(88,166,255,.22);color:var(--acc)}
.seg{display:inline-flex;gap:0;border:1px solid var(--bd);border-radius:8px;overflow:hidden;margin-left:4px}
.segbtn{background:transparent;border:0;border-right:1px solid var(--bd);color:var(--dim);font:inherit;font-size:11px;
 font-weight:600;letter-spacing:.02em;padding:3px 10px;cursor:pointer}
.segbtn:last-child{border-right:0}
.segbtn:hover{background:var(--hover)}
.segbtn.on{background:rgba(63,185,80,.22);color:var(--grn)}
.segbtn.on[data-mode=full_remote]{background:rgba(210,153,34,.25);color:var(--amb)}
.segbtn.on[data-mode=full_local]{background:rgba(88,166,255,.22);color:var(--acc)}
button.ghost{background:transparent;color:var(--dim);border:1px solid var(--bd);font-weight:500;padding:4px 10px;font-size:12px;border-radius:6px;cursor:pointer}
button.ghost:hover{color:var(--fg);background:var(--hover)}
button.ghost.on{color:var(--blu);border-color:var(--blu)}
button.primary{background:var(--blu);color:#04101f;border:0;border-radius:6px;padding:7px 16px;font-weight:600;cursor:pointer;font-size:13px}
button.primary:hover{opacity:.92}
.consequence{width:100%;font-size:11.5px;color:var(--dim);padding:2px 0 0}

/* auth badge */
.authbadge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600;cursor:default}
.authbadge.on{background:rgba(63,185,80,.15);color:var(--grn)}
.authbadge.off{background:rgba(248,81,73,.15);color:var(--red)}

/* sub-header row: subtitle + glossary toggle */
.subrow{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;font-size:12px;margin:10px 0 12px}

/* glossary */
#glossary{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:12.5px}
#glossary dl{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:6px 18px;margin:6px 0 0}
#glossary dt{color:var(--fg);font-weight:600;display:inline}
#glossary dd{color:var(--dim);display:inline;margin:0 0 0 4px}
#glossary .row{margin:0}

/* global banner (errors + backoff + estate alert, consolidated) */
#banner{display:flex;flex-direction:column;gap:6px;margin:10px 0}
.bnln{padding:8px 12px;border-radius:8px;font-size:12.5px;display:flex;gap:8px;align-items:baseline}
.bnln.err{background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.4)}
.bnln.warn{background:rgba(210,153,34,.12);border:1px solid rgba(210,153,34,.4)}
.bnln b.src{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--dim);flex:none}

/* right-now summary strip */
#now_summary{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:baseline;background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 16px;margin:12px 0;font-size:13px}
#now_summary b{font-size:17px;font-variant-numeric:tabular-nums}
#now_summary .lbl{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;margin-left:4px}
#longest{font-size:12px;color:var(--amb);margin-top:2px}

/* tab bar */
.tabbar{display:flex;gap:4px;border-bottom:1px solid var(--bd);margin:6px 0 14px;flex-wrap:wrap}
.tabbtn{background:transparent;border:0;border-bottom:2px solid transparent;color:var(--dim);font:inherit;font-size:13.5px;font-weight:600;padding:8px 14px 9px;cursor:pointer;margin-bottom:-1px}
.tabbtn:hover{color:var(--fg)}
.tabbtn.on{color:var(--fg);border-bottom-color:var(--blu)}
.tabbtn .n{background:#21262d;border-radius:9px;padding:0 6px;font-size:10.5px;margin-left:6px;color:var(--dim)}
.tabpane{display:none}
.tabpane.on{display:block}

/* native tooltips on every metric label + knob (secondary reinforcement; text is never hover-only) */
abbr[title]{text-decoration:none;border-bottom:1px dotted var(--dim);cursor:help}

/* live-health strip */
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;min-width:0}
.tile .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:22px;font-weight:600;margin-top:2px;line-height:1.2}
.tile .v small{font-size:12px;color:var(--dim);font-weight:400}
.tile .sub{color:var(--dim);font-size:11.5px;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* collapsible sections (still used inside tabs for optional/secondary content) */
details.sec{background:var(--card);border:1px solid var(--bd);border-radius:10px;margin:10px 0;overflow:hidden}
details.sec>summary{list-style:none;padding:11px 14px;cursor:pointer;display:flex;align-items:center;gap:10px;font-size:14px;font-weight:600;color:var(--fg);user-select:none;flex-wrap:wrap}
details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary::after{content:'\25B8';margin-left:auto;color:var(--dim);font-size:14px;transition:transform .12s}
details.sec[open]>summary::after{transform:rotate(90deg)}
details.sec>summary .sub{color:var(--dim);font-size:12px;font-weight:400}
details.sec>.body{padding:0 14px 14px;border-top:1px solid var(--bd);padding-top:12px}
h2.h{font-size:14px;font-weight:600;margin:18px 0 8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
h2.h .sub{color:var(--dim);font-size:12px;font-weight:400}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;margin:10px 0}

/* meter/bar */
.bar{height:8px;border-radius:4px;background:#21262d;overflow:hidden;display:flex;margin-top:8px}
.bar i{display:block;height:100%}
.bl{background:var(--grn)}.br{background:var(--amb)}.bblu{background:var(--blu)}

/* tables */
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--dim);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--bd);white-space:nowrap}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--fg)}
th.sortable .arrow{display:inline-block;width:9px;color:var(--blu);font-size:10px}
td{padding:6px 8px;border-bottom:1px solid #21262d;white-space:nowrap}
@media (prefers-color-scheme:light){td{border-bottom-color:#e7ebee}}

/* tags + badges */
.tag{padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
.tag.local{background:rgba(63,185,80,.15);color:var(--grn)}
.tag.remote{background:rgba(210,153,34,.15);color:var(--amb)}
.tag.held{background:rgba(88,166,255,.15);color:var(--blu)}
.tag.reject{background:rgba(248,81,73,.15);color:var(--red)}
.badge{display:inline-block;font-size:11px;padding:0 6px;border-radius:8px;background:#21262d;color:var(--dim);margin-left:6px;vertical-align:1px;font-weight:500}
.badge.ok{color:var(--grn);background:rgba(63,185,80,.12)}
.badge.warn{color:var(--amb);background:rgba(210,153,34,.14)}
.badge.run{color:var(--blu);background:rgba(88,166,255,.14)}
.badge.err{color:var(--red);background:rgba(248,81,73,.14)}
.badge.kind{color:var(--acc);background:rgba(163,113,247,.15);text-transform:uppercase;font-size:9.5px;letter-spacing:.03em}

/* task rows */
.tasks{display:flex;flex-direction:column}
.task{display:grid;grid-template-columns:12px minmax(0,1fr) auto;gap:10px;align-items:start;padding:6px 8px;border-radius:6px;text-decoration:none;color:inherit}
a.task:hover,.task.hov:hover{background:var(--hover)}
.sq{width:9px;height:9px;border-radius:2px;margin-top:6px;background:#484f58;display:inline-block;flex:none}
.sq.run{background:var(--blu);animation:sqpulse 1.3s ease-in-out infinite}
@keyframes sqpulse{0%,100%{opacity:1}50%{opacity:.25}}
.sq.done{background:var(--grn)}.sq.blocked{background:var(--red)}
.sq.stale{background:var(--amb)}
.sq.queued{background:transparent;border:1.5px solid var(--dim)}.sq.empty{background:#30363d}
.tname{font-weight:600;line-height:1.35;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;word-break:break-word}
.tstat{color:var(--dim);font-size:12.5px;margin-top:2px;word-break:break-word;white-space:normal}
.tmeta{color:var(--dim);font-size:12px;white-space:nowrap;text-align:right;line-height:1.35;padding-top:1px}
.tmeta b{color:var(--fg);font-weight:600}
.role{color:var(--fg)}
.rawid{color:var(--dim);font-size:11px}

/* group headers inside a section */
.thead{display:flex;align-items:center;gap:8px;margin:14px 0 4px;color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;flex-wrap:wrap}
.thead .n{background:#21262d;border-radius:9px;padding:0 7px;font-size:11px;color:var(--fg)}
.thead .lg{text-transform:none;letter-spacing:0;font-weight:400}
.empty{color:var(--dim);font-size:12.5px;padding:8px 4px}

/* pagination */
.pager{display:inline-flex;align-items:center;gap:6px;color:var(--dim);font-size:12px;margin-left:auto}
.pager button{background:transparent;color:var(--dim);border:1px solid var(--bd);border-radius:6px;padding:2px 9px;font-size:12px;cursor:pointer;line-height:1.4}
.pager button:hover:not(:disabled){color:var(--fg);background:var(--hover)}
.pager button:disabled{opacity:.35;cursor:not-allowed}
.pager .lbl{color:var(--dim);min-width:44px;text-align:center;font-variant-numeric:tabular-nums}

/* tools row */
.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:4px 0 8px}
.tools input[type=text]{background:var(--bg);border:1px solid var(--bd);color:var(--fg);border-radius:6px;padding:5px 9px;font-size:12.5px;min-width:180px}
.tools label{display:flex;align-items:center;gap:6px;color:var(--dim);font-size:12px}
.tools select{background:var(--bg);border:1px solid var(--bd);color:var(--fg);border-radius:6px;padding:3px 7px;font-size:12px}
.chip{background:transparent;border:1px solid var(--bd);color:var(--dim);border-radius:14px;padding:3px 11px;font-size:12px;cursor:pointer;font-weight:600}
.chip:hover{color:var(--fg);background:var(--hover)}
.chip.on{background:rgba(88,166,255,.16);color:var(--blu);border-color:var(--blu)}

/* config form */
form.cfg .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px 14px}
form.cfg label{display:flex;flex-direction:column;gap:3px}
form.cfg input,form.cfg select{background:var(--bg);border:1px solid var(--bd);color:var(--fg);border-radius:6px;padding:6px 9px;font-size:13px;font-family:inherit}
form.cfg input:focus,form.cfg select:focus{outline:none;border-color:var(--blu)}
form.cfg .k{color:var(--fg);font-size:12.5px;font-weight:500}
form.cfg .hint{color:var(--dim);font-size:11px;font-weight:400}
.fieldflash{animation:flash 1.6s ease-out}
@keyframes flash{0%{background:rgba(88,166,255,.25)}100%{background:transparent}}

/* sparklines */
.spark{width:100%;height:56px;display:block;overflow:visible}
.spark polyline{fill:none;stroke-width:1.6;vector-effect:non-scaling-stroke}
.s1{stroke:var(--blu)}.s2{stroke:var(--amb)}.s3{stroke:var(--grn)}
.spark-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-top:8px}
.spark-cell{background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:10px 12px}
.telemwrap{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px;gap:6px;flex-wrap:wrap}
.telemwrap .legend{font-size:11px;color:var(--dim)}
.l1{color:var(--blu)}.l2{color:var(--amb)}.l3{color:var(--grn)}
.sparkfoot{font-size:11px;color:var(--dim);margin-top:3px}

/* details.tail (used inside task rows) */
details.tail summary{cursor:pointer;color:var(--dim);font-size:12px;list-style:none}
details.tail summary::-webkit-details-marker{display:none}
details.tail pre{margin:6px 0 0;font-size:11.5px;color:var(--dim);white-space:pre-wrap;background:var(--bg);border:1px solid var(--bd);border-radius:6px;padding:8px}

/* windows cards */
.wcard{background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:10px 12px;margin-bottom:8px}
.armwrap{overflow-x:auto}
.armtbl{margin:6px 0 0;width:100%}
.armtbl th,.armtbl td{padding:3px 6px;font-size:11.5px;white-space:nowrap}

/* reason -> knob explainer rows */
.reasonrow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:baseline;padding:8px 0;border-bottom:1px solid #21262d}
@media (prefers-color-scheme:light){.reasonrow{border-bottom-color:#e7ebee}}
.reasonrow:last-child{border-bottom:0}
.reasonrow .rtitle{font-weight:600}
.reasonrow .rwhy{color:var(--dim);font-size:12.5px;margin-top:2px}
.reasonrow .rfix{color:var(--acc);font-size:12px;margin-top:3px}
.reasonrow .rcount{text-align:right;white-space:nowrap}
.reasonrow .rcount b{font-size:16px}
.rbar{height:5px;border-radius:3px;background:#21262d;margin-top:6px;overflow:hidden}
.rbar i{display:block;height:100%;background:var(--amb)}

/* mobile */
@media(max-width:640px){
  .wrap{padding:0 12px 32px}
  header.top{margin:0 -12px;padding:10px 12px}
  header h1{font-size:14px}
  details.sec>summary{padding:11px 12px;font-size:13.5px}
  details.sec>.body{padding:0 12px 12px;padding-top:12px}
  .tmeta{font-size:11.5px}
  .tile .v{font-size:19px}
  .strip{grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}
  .pager{margin-left:0;margin-top:4px}
  .reasonrow{grid-template-columns:1fr}
  .reasonrow .rcount{text-align:left}
}
</style></head><body><div class=wrap>

<header class=top>
 <h1>
  <span id=live class="dot down" title="Overall gateway status. Green = live, red = last poll failed."></span>
  vLLM Gateway
  <span id=modebadge class=stamp title="Routing mode">...</span>
  <span id=modeseg class=seg>
   <button class=segbtn data-mode=local_first>LOCAL FIRST</button>
   <button class=segbtn data-mode=full_remote>FULL REMOTE</button>
   <button class=segbtn data-mode=full_local>FULL LOCAL</button>
  </span>
  <span id=authbadge class=authbadge title="Whether POST /gateway/config and POST /gateway/models/local require the X-Admin-Token header.">...</span>
 </h1>
 <span class=rz id=updated style="margin-left:auto;font-size:12px">connecting...</span>
 <div class=consequence id=modeconsequence></div>
</header>

<div class=subrow>
 <span class=rz><abbr title="This gateway listens on :8000 and forwards to the local vLLM engine on :8001. It queues, serialises, and (when local is full) overflows to a paid remote provider.">:8000 capacity-routing gateway</abbr> &rarr; local engine :8001 &middot; overflow provider <b id=rm class=mono>--</b> &middot; <a href=/gateway/aliases/page>custom aliases</a></span>
 <button class=chip id=glossary_btn type=button>? Glossary</button>
</div>

<div id=glossary hidden>
 <b>Glossary</b> &mdash; every term used on this page, in one place (not hover-only).
 <dl>
  <div class=row><dt>Lane</dt><dd>one concurrent request slot on the local engine. "Budget" = how many lanes exist.</dd></div>
  <div class=row><dt>TTFT</dt><dd>time to first token &mdash; how long before streaming starts.</dd></div>
  <div class=row><dt>TPOT / inter-token latency</dt><dd>seconds between output tokens once streaming has started.</dd></div>
  <div class=row><dt>KV-cache %</dt><dd>how full the engine's attention-cache memory is.</dd></div>
  <div class=row><dt>ctx</dt><dd>context &mdash; total prompt+output tokens a request occupies while in flight.</dd></div>
  <div class=row><dt>prefill</dt><dd>the engine reading your prompt, before it can generate the first token.</dd></div>
  <div class=row><dt>OOM backoff</dt><dd>the local engine ran out of memory; the gateway holds off sending it new work for a bit.</dd></div>
  <div class=row><dt>Serialize-solo</dt><dd>a request big enough that it claims the *entire* lane budget alone.</dd></div>
  <div class=row><dt>Tiny fast-lane</dt><dd>very small requests get their own reserved headroom so they never queue behind big ones.</dd></div>
  <div class=row><dt>Background</dt><dd>a request tagged (by user-agent or IP) as non-interactive &mdash; cron/batch traffic that waits less patiently and never blocks a human's turn.</dd></div>
  <div class=row><dt>FULL REMOTE / FULL LOCAL</dt><dd>routing modes &mdash; see the tooltip on the mode buttons above.</dd></div>
  <div class=row><dt>p50 / p95</dt><dd>median, then the worst 5% &mdash; a shorthand for "typical" vs "bad case".</dd></div>
  <div class=row><dt>Evalkit score</dt><dd>an automated quality check on an engine-benchmark window's output, out of 45.</dd></div>
  <div class=row><dt>Arm</dt><dd>one configuration variant being A/B-tested in a frontier-queue benchmark window.</dd></div>
  <div class=row><dt>Pool</dt><dd>the KV-cache block pool size an arm was tested with.</dd></div>
  <div class=row><dt>MTP</dt><dd>multi-token prediction &mdash; the engine's speculative-decoding scheme.</dd></div>
  <div class=row><dt>Claims</dt><dd>research-job output: how many extracted claims survived verification, out of how many drafted.</dd></div>
  <div class=row><dt>Degraded</dt><dd>a research job finished without enough usable evidence to trust.</dd></div>
 </dl>
</div>

<div id=banner></div>

<!-- Live health strip: always visible, whichever tab is open -->
<div class=strip id=strip></div>

<div class=tabbar id=tabbar>
 <button class="tabbtn on" data-tab=now>Now <span class=n id=tab_n_now>&middot;</span></button>
 <button class=tabbtn data-tab=traffic>Traffic <span class=n id=tab_n_traffic>&middot;</span></button>
 <button class=tabbtn data-tab=settings>Settings</button>
</div>

<!-- ================= NOW ================= -->
<div class="tabpane on" id=pane-now>

 <div id=now_summary>loading&hellip;</div>
 <div id=longest hidden></div>

 <div class=tools>
  <span class=chip id=k_all data-k="" style="border-color:var(--blu);color:var(--blu)">All</span>
  <span class=chip data-k=request>Requests</span>
  <span class=chip data-k=lane>Agent lanes</span>
  <span class=chip data-k=window>Windows</span>
  <span class=chip data-k=research>Research</span>
  <label><input type=checkbox id=n_showdone> include finished</label>
  <span class=rz style="font-size:11.5px" id=n_state_legend></span>
  <span class=pager>
   <button id=n_prev disabled>&lsaquo;</button>
   <span class=lbl id=n_page>1/1</span>
   <button id=n_next disabled>&rsaquo;</button>
  </span>
 </div>
 <div class=tw>
  <table id=n_table>
   <thead><tr>
    <th></th>
    <th class=sortable data-sort=kind>kind<span class=arrow></span></th>
    <th class=sortable data-sort=who>who / what<span class=arrow></span></th>
    <th>status</th>
    <th class=sortable data-sort=age>age<span class=arrow></span></th>
    <th></th>
   </tr></thead>
   <tbody id=n_rows><tr><td colspan=6 class=empty>loading...</td></tr></tbody>
  </table>
 </div>
 <div id=n_err style="color:var(--red);font-size:12px;margin-top:6px"></div>
 <div style="margin-top:6px"><button class="ghost" id=n_csv type=button>Export CSV</button></div>

 <details class=sec id=sec-health>
  <summary>Estate watchdog <span class=sub>separate service on 10.0.1.10 &middot; not part of this gateway &middot; refreshes every 2 s</span></summary>
  <div class=body>
   <div id=health_alert hidden style="margin:0 0 10px;padding:8px 12px;border-radius:8px;background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.4);color:var(--fg);font-size:13px"></div>
   <div class=thead>Checks <span class=n id=n_health>...</span><span class="lg rz" id=health_line>loading...</span></div>
   <div id=t_health style="display:flex;flex-wrap:wrap;gap:6px 14px;padding:4px 0 6px;font-size:12.5px"><span class=empty>loading...</span></div>
  </div>
 </details>

</div>

<!-- ================= TRAFFIC ================= -->
<div class=tabpane id=pane-traffic>

 <h2 class=h>Routing mix <span class=sub>local vs. paid overflow, and exactly why each overflow happened</span></h2>
 <div class=card>
  <div class=bar id=lrbar></div>
  <div id=lrtxt class=rz style="margin-top:8px;font-size:12.5px"></div>
 </div>

 <div class=card>
  <div class=thead>Why requests went to the paid provider <span class="lg rz">ranked by how often &middot; each links to the setting that causes it</span></div>
  <div id=reasons><span class=empty>loading...</span></div>
  <div class=thead style="margin-top:14px">Outcome legend</div>
  <div class="lg rz" style="font-size:12px;line-height:2">
   <span class="tag local">local</span> served by the local engine &nbsp;&middot;&nbsp;
   <span class="tag remote">remote</span> sent to the paid overflow provider &nbsp;&middot;&nbsp;
   <span class="tag held">held</span> background traffic paused, waiting for local to recover (never billed) &nbsp;&middot;&nbsp;
   <span class="tag reject">rejected</span> background traffic refused outright (a deliberate maintenance window, no wait, no bill)
  </div>
 </div>

 <h2 class=h>Telemetry <span class=sub>engine internals, GPU/host trends, per-client rollups &middot; live, refreshes every 2 s</span></h2>
 <div class=card>
  <div class=strip id=telem_tiles></div>
  <div class=spark-grid>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="Prompt tokens/s (input) and generation tokens/s (output) as reported by the engine's Prometheus metrics.">Tokens/s (prompt / generation)</abbr></div><div class=legend><span class=l1>&#9632;</span> prompt <span class=l2>&#9632;</span> gen</div></div><svg class=spark id=sp_toks viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_toks></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="Time to first token, seconds. p50 = median wait before streaming starts; p95 = worst 5%.">TTFT p50 / p95 (s)</abbr></div><div class=legend><span class=l1>&#9632;</span> p50 <span class=l2>&#9632;</span> p95</div></div><svg class=spark id=sp_ttft viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_ttft></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="Inter-token latency, seconds. Gap between consecutive output tokens after the first -- streaming smoothness.">Inter-token latency (s)</abbr></div><div class=legend><span class=l1>&#9632;</span> p50 <span class=l2>&#9632;</span> p95</div></div><svg class=spark id=sp_tpot viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_tpot></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="KV-cache utilisation % (engine) and gateway lanes currently in flight.">KV-cache % / lanes</abbr></div><div class=legend><span class=l1>&#9632;</span> KV% <span class=l2>&#9632;</span> lanes</div></div><svg class=spark id=sp_kv viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_kv></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="GPU 0 utilisation %, temperature C, and power draw as % of card cap.">GPU 0 &mdash; util / temp / power</abbr></div><div class=legend><span class=l1>&#9632;</span> util% <span class=l2>&#9632;</span> temp&deg;C <span class=l3>&#9632;</span> pwr%cap</div></div><svg class=spark id=sp_gpu0 viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_gpu0></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="GPU 1 utilisation %, temperature C, and power draw as % of card cap.">GPU 1 &mdash; util / temp / power</abbr></div><div class=legend><span class=l1>&#9632;</span> util% <span class=l2>&#9632;</span> temp&deg;C <span class=l3>&#9632;</span> pwr%cap</div></div><svg class=spark id=sp_gpu1 viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_gpu1></div></div>
   <div class=spark-cell><div class=telemwrap><div class=k><abbr title="Percentage of recent requests routed to the paid overflow provider.">Remote-overflow share %</abbr></div><div class=legend></div></div><svg class=spark id=sp_remote viewBox="0 0 300 56" preserveAspectRatio=none></svg><div class=sparkfoot id=sf_remote></div></div>
   <div class=spark-cell><div class=k><abbr title="Whether the shim's periodic scrape of the engine's Prometheus /metrics is succeeding.">Engine metrics scrape</abbr></div><div class=v style="font-size:14px;margin-top:6px" id=telem_engok>...</div></div>
  </div>
 </div>

 <h2 class=h>Per-client usage <span class=sub>who is calling this gateway, and how much</span></h2>
 <div class=card>
  <div class=tools>
   <span class=rz style="font-size:12px">host key: <span id=hostkey></span></span>
   <label>window <select id=pc_window><option value=uptime selected>since gateway start</option><option value=day>last 24h</option></select></label>
   <span class=rz style="font-size:11.5px" id=pc_filtered_note></span>
   <span class=pager>
    <button id=pc_prev disabled>&lsaquo;</button>
    <span class=lbl id=pc_page>1/1</span>
    <button id=pc_next disabled>&rsaquo;</button>
   </span>
  </div>
  <div class=tw><table><thead><tr>
   <th class=sortable data-sort=client>client<span class=arrow></span></th>
   <th class=sortable data-sort=requests>requests<span class=arrow></span></th><th>local</th><th>remote</th>
   <th><abbr title="Total output tokens served (exact where the engine reported them; else lower-bound from chunk count, marked ~).">tokens out</abbr></th>
   <th><abbr title="Average seconds spent waiting for a free lane before starting.">avg wait</abbr></th>
   <th><abbr title="Average time to first token, seconds.">avg TTFT</abbr></th>
   <th>errors</th>
   <th class=sortable data-sort=cost><abbr title="Rough USD estimate for what the remote-overflow calls have cost (uptime window only).">est. cost</abbr><span class=arrow></span></th>
  </tr></thead><tbody id=telem_clients><tr><td colspan=9 class=empty>loading...</td></tr></tbody></table></div>
 </div>

 <h2 class=h>Error feed <span class=sub>failovers and non-2xx completions, newest first</span></h2>
 <div class=card>
  <div class=pager style="margin-bottom:6px">
   <button id=ef_prev disabled>&lsaquo;</button>
   <span class=lbl id=ef_page>1/1</span>
   <button id=ef_next disabled>&rsaquo;</button>
  </div>
  <div class=tw><table><thead><tr>
   <th>time</th><th>client</th><th>endpoint</th>
   <th><abbr title="Route the request took: local, remote (overflow), held, or rejected.">route</abbr></th>
   <th>reason</th><th>status</th>
  </tr></thead><tbody id=telem_errors><tr><td colspan=6 class=empty>loading...</td></tr></tbody></table></div>
 </div>

 <h2 class=h>Request history <span class=sub>on-disk log, survives restarts &middot; server-side paginated</span></h2>
 <div class=card>
  <div class=tools>
   <input type=text id=h_client placeholder="filter by client...">
   <input type=text id=h_route placeholder="filter by route/reason (local, remote, tiny...)">
   <label>per page <select id=h_limit>
    <option>10</option><option selected>25</option><option>50</option><option>100</option>
   </select></label>
   <span class=rz id=h_logstate></span>
   <span class=pager>
    <button id=h_prev disabled>&lsaquo;</button>
    <span class=lbl id=h_page>1</span>
    <button id=h_next disabled>&rsaquo;</button>
   </span>
  </div>
  <div class=tw><table><thead><tr>
   <th>time</th><th>client</th><th>route</th>
   <th><abbr title="First few chars of the user prompt.">preview</abbr></th>
   <th>in&rarr;out</th>
   <th><abbr title="Time to first token, seconds.">TTFT</abbr></th>
   <th>duration</th><th>status</th>
  </tr></thead><tbody id=h_rows><tr><td colspan=8 class=empty>loading...</td></tr></tbody></table></div>
 </div>

 <h2 class=h>Recent requests <span class=sub>in-memory ring buffer, this process's uptime only &middot; &#9889; = streamed</span></h2>
 <div class=card>
  <div class=tools><span class="lg rz" id=req_summary></span>
   <span class=pager>
    <button id=ev_prev disabled>&lsaquo;</button>
    <span class=lbl id=ev_page>1/1</span>
    <button id=ev_next disabled>&rsaquo;</button>
   </span>
  </div>
  <div class=tw><table><thead><tr>
   <th>time</th><th>endpoint</th><th>client</th>
   <th>route</th><th>reason</th>
   <th><abbr title="Prompt tokens in -> max output tokens requested. streamed = lightning icon.">size (in&rarr;out)</abbr></th>
   <th>waited</th>
  </tr></thead><tbody id=ev><tr><td colspan=7 class=empty>loading...</td></tr></tbody></table></div>
 </div>

</div>

<!-- ================= SETTINGS ================= -->
<div class=tabpane id=pane-settings>

 <h2 class=h>Local model <span class=sub>switch the served checkpoint &middot; engine restarts &middot; overflow covers the gap</span></h2>
 <div class=card>
  <div class=rz style="font-size:12.5px">Scans <span id=mdir class=mono></span>. Any HF checkpoint works. Switching restarts the engine (~40 s warm / 4-5 min cold); traffic falls back to remote overflow until it is healthy.</div>
  <div class=tw><table><thead><tr><th></th><th>model</th><th>size</th><th>format</th><th></th></tr></thead><tbody id=lm></tbody></table></div>
  <div style="margin-top:8px"><span id=lmmsg class=rz></span></div>
 </div>

 <h2 class=h>Admin access <span class=sub>who can change settings below</span></h2>
 <div class=card id=authcard>
  <div id=authtext class=rz style="font-size:12.5px"></div>
  <div style="margin-top:8px"><button class=ghost id=forget_token type=button>Forget saved admin token in this browser</button></div>
 </div>

 <h2 class=h>Provider &amp; routing settings <span class=sub>applied live, saved to shim.env</span></h2>
 <div class=card>
 <form class=cfg autocomplete=off>

  <div class=thead>Overflow provider &mdash; where paid requests go</div>
  <label class=k>Provider preset
   <select id=f_preset>
    <option value="">-- pick to autofill base + model --</option>
    <option value="https://api.deepseek.com|deepseek-v4-flash">DeepSeek v4-flash</option>
    <option value="https://api.deepseek.com|deepseek-v4-pro">DeepSeek v4-pro</option>
    <option value="https://api.minimax.io/v1|MiniMax-M3">MiniMax M3</option>
    <option value="https://dashscope-intl.aliyuncs.com/compatible-mode/v1|qwen3-coder-next">Qwen3-Coder-Next (DashScope intl)</option>
    <option value="https://dashscope-intl.aliyuncs.com/compatible-mode/v1|qwen3.7-flash">qwen3.7-flash (DashScope intl)</option>
    <option value="https://openrouter.ai/api/v1|minimax/minimax-m3">OpenRouter &rarr; MiniMax M3</option>
   </select>
  </label>
  <div class=row>
   <label><span class=k>Overflow provider URL</span><span class=hint>base URL of the OpenAI-compatible endpoint</span>
    <input id=f_remote_base placeholder=https://api.minimax.io/v1></label>
   <label><span class=k>Overflow model name</span><span class=hint>model id that provider expects on /v1/chat/completions</span>
    <input id=f_remote_model placeholder=MiniMax-M3></label>
   <label><span class=k>Overflow API key</span><span class=hint id=keystate>&nbsp;</span>
    <input id=f_remote_key type=password placeholder="leave blank to keep current"></label>
   <label><span class=k>Send everything to the overflow provider?</span><span class=hint>1 = yes, bypass the local engine entirely &middot; 0 = local-first (normal)</span>
    <input id=f_force_remote type=number min=0 max=1></label>
  </div>

  <div class=thead>Local capacity &mdash; how much the local engine can take at once</div>
  <div class=row>
   <label><span class=k>How many requests can run locally at once?</span><span class=hint>concurrent lanes &middot; 1 = strictly one at a time</span>
    <input id=f_local_budget type=number min=1 max=8></label>
   <label><span class=k>How long should a request wait for a free lane?</span><span class=hint>seconds &middot; used for background during peak hours, and for interactive only if "never overflow" below is off</span>
    <input id=f_local_wait_secs type=number min=0 step=1></label>
   <label><span class=k>Should interactive traffic ever overflow while waiting?</span><span class=hint>1 = never, it queues until a lane is free (default) &middot; 0 = restores the wait-above-then-overflow behaviour</span>
    <input id=f_interactive_never_overflow type=number min=0 max=1></label>
   <label><span class=k>Prompt and output tokens all lanes may reserve</span><span class=hint>estimated tokens &middot; admission memory limit</span>
    <input id=f_token_budget type=number min=0 step=50000></label>
   <label><span class=k>After an out-of-memory crash, how long to back off?</span><span class=hint>seconds before probing the local engine again</span>
    <input id=f_oom_backoff_secs type=number min=0 step=10></label>
  </div>

  <div class=thead>Routing guards &mdash; when a request skips the local engine</div>
  <div class=row>
   <label><span class=k>Send to overflow if requested output is at least&hellip;</span><span class=hint>tokens &middot; caused reason "big-out"</span>
    <input id=f_big_output type=number min=0 step=1000></label>
   <label><span class=k>Send to overflow if the prompt is at least&hellip;</span><span class=hint>tokens, 0 = off &middot; caused reason "big-prompt"</span>
    <input id=f_big_prompt type=number min=0 step=1000></label>
   <label><span class=k>Hard size cap: prompt + output above this always goes to overflow</span><span class=hint>tokens &middot; caused reason "size"</span>
    <input id=f_max_local_tokens type=number min=0 step=10000></label>
   <label><span class=k>Clamp any local request's output to at most&hellip;</span><span class=hint>tokens</span>
    <input id=f_local_max_out type=number min=0 step=1024></label>
   <label><span class=k>Above this size, a request's lane cost scales with its size</span><span class=hint>tokens &middot; below this, every request costs exactly 1 lane</span>
    <input id=f_big_tokens type=number min=0 step=1000></label>
   <label><span class=k>How many tokens equal one lane, for a big request?</span><span class=hint>tokens/unit &middot; e.g. a 100K-token request costs ceil(100000&divide;this) lanes, capped by the budget</span>
    <input id=f_tokens_per_unit type=number min=1000 step=1000></label>
   <label><span class=k>Charge lanes for PREDICTED computed tokens instead of raw prompt size?</span><span class=hint>0 = off, default &middot; 1 = a repeated-prefix turn with a small new suffix costs ~1 lane instead of its full size &middot; check the mis-estimate feed before flipping this on</span>
    <input id=f_use_computed_cost type=number min=0 max=1></label>
   <label><span class=k>Safety margin added to a predicted-cheap request's cost</span><span class=hint>tokens &middot; padding for tokenizer/cache-boundary slop</span>
    <input id=f_prefix_hit_margin_tokens type=number min=0 step=128></label>
   <label><span class=k>Prefill admission window (cache-aware mode)</span><span class=hint>seconds of uncached prefill the engine may hold in its queue &middot; a request that does not fit waits for a lane, then overflows &middot; reason "prefill"</span>
    <input id=f_prefill_admit_secs type=number min=0 step=5></label>
   <label><span class=k>A request this small always fits the window</span><span class=hint>seconds of prefill</span>
    <input id=f_light_prefill_secs type=number min=0 step=1></label>
   <label><span class=k>Monster = this many seconds of prefill already in flight (cache-aware mode)</span><span class=hint>seconds &middot; new arrivals go remote &middot; reason "monster"</span>
    <input id=f_monster_prefill_secs type=number min=0 step=5></label>
   <label><span class=k>Treat this much in-flight context as "a monster is running"</span><span class=hint>tokens &middot; caused reason "monster"</span>
    <input id=f_monster_inflight type=number min=0 step=10000></label>
   <label><span class=k>Upper bound on the first-token wait budget</span><span class=hint>seconds</span>
    <input id=f_first_token_max type=number min=5 step=5></label>
   <label><span class=k>Estimated prefill speed</span><span class=hint>tokens/s, used to size first-token deadlines</span>
    <input id=f_prefill_tps type=number min=100 step=100></label>
  </div>

  <div class=thead>Priority lanes &amp; behaviour</div>
  <div class=row>
   <label><span class=k>Treat requests this small as "tiny"</span><span class=hint>tokens &middot; tiny requests get their own fast lane</span>
    <input id=f_tiny_tokens type=number min=0 step=100></label>
   <label><span class=k>Extra lanes reserved just for tiny requests</span><span class=hint>beyond the normal budget &middot; caused reason "tiny-fast" when full</span>
    <input id=f_tiny_extra_lanes type=number min=0 max=4></label>
   <label><span class=k>Lanes always kept free for interactive (non-background) traffic</span><span class=hint>caused reason "bg-yield"</span>
    <input id=f_fg_reserved type=number min=0 max=4></label>
   <label><span class=k>How long background traffic waits for a lane</span><span class=hint>seconds, before overflowing</span>
    <input id=f_bg_wait_secs type=number min=0 step=1></label>
   <label><span class=k>User-agent substrings that mark a request "background"</span><span class=hint>pipe-separated, e.g. scheduled|cron|job</span>
    <input id=f_bg_markers placeholder="scheduled cron job"></label>
   <label><span class=k>Peak hours (UTC) &mdash; bias background traffic to wait for local</span><span class=hint>comma ranges, e.g. 1-4,6-10</span>
    <input id=f_peak_hours_utc></label>
   <label><span class=k>Strip /think for background traffic?</span><span class=hint>0/1</span>
    <input id=f_bg_no_think type=number min=0 max=1></label>
   <label><span class=k>Client IPs that always get /think stripped</span><span class=hint>comma-separated</span>
    <input id=f_no_think_ips placeholder="10.0.1.10,10.0.1.250"></label>
   <label><span class=k>Log every request body to disk?</span><span class=hint>0/1 &middot; needed for the History tab</span>
    <input id=f_log_requests type=number min=0 max=1></label>
  </div>

  <div style="margin-top:12px">
   <button type=button id=save class=primary>Save settings</button>
   <span id=savemsg class=rz></span>
  </div>
 </form>
 </div>

</div>

</div>
<script>
(function(){
  // ---- shared helpers ----
  const $=s=>document.querySelector(s);
  const $$=s=>document.querySelectorAll(s);
  const esc=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  // ONE family of time formatters, consistently suffixed, so the page never mixes "14s elapsed" /
  // "0s ago" / "48m ago" / "up 719h25m" styles for the same underlying quantity again.
  const ago=s=>s==null?'?':s<60?Math.round(s)+'s ago':s<5400?Math.round(s/60)+'m ago':s<172800?Math.round(s/3600)+'h ago':Math.round(s/86400)+'d ago';
  const fmtAgo=t=>{const s=Math.max(0,Date.now()/1000-t);return s<60?s.toFixed(0)+'s ago':(s/60).toFixed(0)+'m ago';};
  const dur=s=>s==null?'--':(s<90?Math.round(s)+'s':s<5400?Math.round(s/60)+'m':(s/3600).toFixed(1)+'h');
  const fmtDur=s=>s==null?'--':(s<1?Math.round(s*1000)+'ms':s<90?s.toFixed(1)+'s':(s/60).toFixed(1)+'m');
  const when=t=>{if(!t)return'';let x=String(t).replace(' ','T');if(!/Z$|[+-]\d\d:\d\d$/.test(x))x+='Z';const d=new Date(x);if(isNaN(d))return esc(t);const s=(Date.now()-d.getTime())/1000;return s<86400?ago(s):d.toLocaleDateString(undefined,{month:'short',day:'numeric'})+' '+d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'});};
  const tile=(k,v,sub)=>`<div class=tile><div class=k>${k}</div><div class=v>${v}</div>${sub?`<div class=sub>${sub}</div>`:''}</div>`;
  const routeTag=r=>r==='local'?'local':r==='held'?'held':r==='rejected-bg'?'reject':'remote';

  // ---- host -> role map (item 42/43): who is actually calling this gateway ----
  const HOSTS={'10.0.1.10':'agents-prod / Hermes','10.0.1.11':'Applicant','10.0.1.12':'ubuntuide01 / pi',
               '10.0.1.225':'this box (local)','10.0.1.250':'Hermes scheduler','127.0.0.1':'this box (local)'};
  function hostRole(ip){for(const k in HOSTS){if(ip&&ip.indexOf(k)===0)return HOSTS[k];}return null;}
  function clientDisplay(name,ip,ua){
    const role=hostRole(ip||name);
    const primary=role?role:esc(name||ip||'?');
    const secondary=[ip&&ip!==primary?esc(ip):null, ua?esc(ua):null].filter(Boolean).join(' &middot; ');
    return '<span class=role>'+primary+'</span>'+(secondary?' <span class=rawid>'+secondary+'</span>':'');
  }
  $('#hostkey').innerHTML=Object.entries(HOSTS).map(([ip,r])=>'<span class=mono>'+ip+'</span>='+esc(r)).join(' &middot; ');

  // ---- admin-token fetch (for mutating endpoints) ----
  function adminToken(){try{return localStorage.getItem('shim_admin_token')||'';}catch(e){return'';}}
  async function adminFetch(url,opts){
    opts=opts||{};opts.headers=Object.assign({'Content-Type':'application/json'},opts.headers||{});
    const t=adminToken();if(t)opts.headers['X-Admin-Token']=t;
    let r=await fetch(url,opts);
    if(r.status===401){
      const entered=prompt('Admin token required for this action (kept in this browser only):');
      if(entered){try{localStorage.setItem('shim_admin_token',entered);}catch(e){}opts.headers['X-Admin-Token']=entered;r=await fetch(url,opts);}
    }
    return r;
  }
  $('#forget_token').addEventListener('click',()=>{try{localStorage.removeItem('shim_admin_token');}catch(e){}$('#forget_token').textContent='forgotten -- next save will re-prompt';});

  // ---- global error/alert banner (item 63): every subsystem writes ONE line here instead of
  // scattering #err/#t_err/#win_err/#telem_engok/#health_alert around the page. ----
  const BANNER={};   // key -> {text, level}
  function setBanner(key,text,level){ // level 'err'|'warn'|null(clear)
    if(!text){delete BANNER[key];}else{BANNER[key]={text,level:level||'err'};}
    renderBanner();
  }
  function renderBanner(){
    const keys=Object.keys(BANNER);
    $('#banner').innerHTML=keys.map(k=>{const b=BANNER[k];
      return '<div class="bnln '+b.level+'"><b class=src>'+esc(k)+'</b><span>'+esc(b.text)+'</span></div>';
    }).join('');
  }

  // ---- generic client-side pagination + sort helper ----
  const PAGE={now:1,pc:1,ef:1,ev:1};
  const SORT={now:{key:null,dir:1},pc:{key:null,dir:-1}};
  function sortArr(arr,key,dir){
    if(!key)return arr;
    return arr.slice().sort((a,b)=>{const av=a[key],bv=b[key];
      if(typeof av==='number'||typeof bv==='number')return((av||0)-(bv||0))*dir;
      return String(av||'').localeCompare(String(bv||''))*dir;});
  }
  function paginate(arr,key,per){
    const total=arr.length,pages=Math.max(1,Math.ceil(total/per));
    if(PAGE[key]>pages)PAGE[key]=pages;
    const p=PAGE[key],from=(p-1)*per,to=Math.min(from+per,total);
    return {slice:arr.slice(from,to),page:p,pages,total,from,to};
  }
  function wirePager(key,prevId,nextId,lblId,pages,onchange){
    const prev=$(prevId),next=$(nextId),lbl=$(lblId);if(!prev||!next||!lbl)return;
    prev.disabled=PAGE[key]<=1;next.disabled=PAGE[key]>=pages;
    lbl.textContent=PAGE[key]+'/'+pages;
    prev.onclick=()=>{if(PAGE[key]>1){PAGE[key]--;onchange();}};
    next.onclick=()=>{if(PAGE[key]<pages){PAGE[key]++;onchange();}};
  }
  function wireSort(tableSel,sortKey,onchange){
    $$(tableSel+' th.sortable').forEach(th=>{
      th.addEventListener('click',()=>{
        const k=th.dataset.sort,s=SORT[sortKey];
        s.dir=(s.key===k)?-s.dir:1;s.key=k;
        $$(tableSel+' th.sortable .arrow').forEach(a=>a.textContent='');
        th.querySelector('.arrow').textContent=s.dir>0?'\u25B2':'\u25BC';
        onchange();
      });
    });
  }
  function csvDownload(filename,rows){
    const esc2=v=>{v=String(v==null?'':v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;};
    const csv=rows.map(r=>r.map(esc2).join(',')).join('\n');
    const blob=new Blob([csv],{type:'text/csv'});
    const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=filename;
    document.body.appendChild(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(a.href),4000);
  }

  // change-detection: skip a full innerHTML re-render when the underlying data hasn't
  // changed (item 58) -- cheap JSON-string compare, keyed per render target.
  const _lastRender={};
  function renderIfChanged(key,data,fn){
    const sig=JSON.stringify(data);
    if(_lastRender[key]===sig)return false;
    _lastRender[key]=sig;fn();return true;
  }

  // Global blink phase kept only as a fallback flag; the actual pulse is a CSS animation now
  // (item 57) so it costs no forced layout, and prefers-reduced-motion turns it off for free.

  // ==================== TAB BAR (items 34/35/50/53/54) ====================
  let TAB='now';
  function applyTab(t,push){
    TAB=t;
    $$('.tabbtn').forEach(b=>b.classList.toggle('on',b.dataset.tab===t));
    $$('.tabpane').forEach(p=>p.classList.toggle('on',p.id==='pane-'+t));
    try{localStorage.setItem('shim_tab',t);}catch(e){}
    if(push)history.replaceState(null,'','#'+t);
  }
  $$('.tabbtn').forEach(b=>b.addEventListener('click',()=>applyTab(b.dataset.tab,true)));
  (function initTab(){
    let t=(location.hash||'').replace('#','');
    if(!['now','traffic','settings'].includes(t)){try{t=localStorage.getItem('shim_tab')||'now';}catch(e){t='now';}}
    applyTab(t,false);
  })();
  function goToSetting(fieldId){
    applyTab('settings',true);
    const el=document.getElementById(fieldId);
    if(el){el.scrollIntoView({block:'center'});el.focus();
      const label=el.closest('label');(label||el).classList.add('fieldflash');
      setTimeout(()=>(label||el).classList.remove('fieldflash'),1700);}
  }

  // ---- glossary (item 24): a real panel, not a hover-only tooltip ----
  $('#glossary_btn').addEventListener('click',()=>{
    const g=$('#glossary');g.hidden=!g.hidden;
    $('#glossary_btn').classList.toggle('on',!g.hidden);
    try{localStorage.setItem('shim_glossary',g.hidden?'0':'1');}catch(e){}
  });
  try{if(localStorage.getItem('shim_glossary')==='1'){$('#glossary').hidden=false;$('#glossary_btn').classList.add('on');}}catch(e){}

  // ==================== TOP + STATS (uses /gateway/stats) ====================
  let modelName='?', FR=0, MODE='local_first', lastOk=0, CFG={};
  async function loadModels(){try{const r=await fetch('/v1/models');const d=await r.json();modelName=(d.data&&d.data[0]&&d.data[0].id)||'?';}catch(e){}}
  const MODE_LABEL={local_first:'LOCAL-FIRST',full_remote:'FULL REMOTE',full_local:'FULL LOCAL (queue, $0)'};
  const MODE_CLASS={local_first:'stamp local',full_remote:'stamp remote',full_local:'stamp locked'};
  const MODE_CONSEQUENCE={local_first:'Requests use the local engine first; only overflow to the paid provider when every lane is busy or a guard fires.',
                          full_remote:'Every completion goes straight to the paid overflow provider -- the local engine sits idle.',
                          full_local:'Never spends money: a request with no free local lane QUEUES for one instead of overflowing.'};
  const MODE_BODY={local_first:{local_only:0,force_remote:0},
                   full_remote:{local_only:0,force_remote:1},
                   full_local:{local_only:1,force_remote:0}};
  function renderMode(){
    const b=$('#modebadge');
    if(b){b.textContent=MODE_LABEL[MODE]||MODE;b.className=MODE_CLASS[MODE]||'stamp';}
    $$('#modeseg .segbtn').forEach(x=>x.classList.toggle('on',x.dataset.mode===MODE));
    $('#modeconsequence').textContent=MODE_CONSEQUENCE[MODE]||'';
  }
  async function setMode(m){
    if(!MODE_BODY[m])return;
    const prev=MODE;
    try{
      const r=await adminFetch('/gateway/config',{method:'POST',body:JSON.stringify(MODE_BODY[m])});
      const d=await r.json();
      MODE=(d.config&&d.config.mode)||m;
    }catch(e){MODE=prev;setBanner('mode','mode change failed: '+e,'err');}
    FR=(MODE==='full_remote')?1:0;renderMode();loadCfg();
  }
  $$('#modeseg .segbtn').forEach(x=>x.addEventListener('click',()=>setMode(x.dataset.mode)));

  setInterval(()=>{const s=lastOk?(Date.now()-lastOk)/1000:null;const lu=$('#updated'),ld=$('#live');if(!lu)return;
    if(s==null){lu.textContent='connecting...';return;}
    lu.textContent='live \u00b7 updated '+Math.round(s)+'s ago';
    ld.className='dot '+(s<8?'up':s<30?'warn':'down');
  },1000);

  let statsData=null,recentEvents=[];
  async function tickStats(){
    let s;try{const r=await fetch('/gateway/stats',{cache:'no-store'});s=await r.json();setBanner('stats',null);lastOk=Date.now();}
    catch(e){setBanner('stats','reconnecting... (showing last data)','warn');return;}
    statsData=s;$('#rm').textContent=s.remote_model||'--';
    const uh=Math.floor(s.uptime/3600),um=Math.floor(s.uptime%3600/60);
    const gpuAvg=(s.gpu||[]).length?Math.round((s.gpu.reduce((a,g)=>a+(+g.util||0),0)/s.gpu.length)):null;
    const modelShort=modelName.replace(/^.*\//,'').slice(0,26);
    const bk=s.backoff>0?`<span class=sub style=color:var(--amb)>OOM backoff ${s.backoff}s</span>`:'';
    setBanner('oom', s.backoff>0?('Local engine is in OOM backoff for another '+s.backoff+'s -- new requests overflow to the paid provider until it clears.'):null, 'warn');
    $('#strip').innerHTML=[
      `<div class=tile><div class=k><abbr title="Whether the local vLLM engine on :8001 is responding to /health.">Local engine</abbr></div><div class=v><span class="dot ${s.local_healthy?'up':'down'}"></span> ${s.local_healthy?'up':'DOWN'}</div><div class=sub title="${esc(modelName)}">${esc(modelShort||'--')}</div></div>`,
      `<div class=tile><div class=k><abbr title="Concurrent request slots (lanes) in use / total available. Waiting shown when requests are queued for one, broken out by class.">Lanes</abbr></div><div class=v class=mono>${s.inflight}<small>/${s.budget}</small>${s.waiting?` <span style=color:var(--amb)>+${s.waiting} waiting${s.waiting_by_class?` (${s.waiting_by_class.interactive||0} interactive, ${s.waiting_by_class.background||0} bg)`:''}</span>`:''}</div><div class=sub><abbr title="Reserved prompt plus bounded output tokens across all lanes / token-budget cap.">context reserved ${((s.inflight_reserved_tokens||0)/1000).toFixed(0)}K / ${((s.token_budget||0)/1000).toFixed(0)}K cap</abbr>${bk?' &middot; '+bk:''}</div></div>`,
      `<div class=tile><div class=k><abbr title="Share of requests served by the local engine vs sent to the overflow provider.">Served local</abbr></div><div class=v class=mono style=color:var(--grn)>${s.local_pct}<small>%</small></div><div class=sub>overflow ${s.remote_pct}% &middot; avg wait ${s.avg_wait}s</div></div>`,
      `<div class=tile><div class=k><abbr title="Live tokens/s from the engine's Prometheus metrics (prompt + generation combined). '--' if the engine is down or not yet scraped.">Tokens/s now</abbr></div><div class=v class=mono id=tps_now>--</div><div class=sub>time to first token <span id=ttft_line>--</span></div></div>`,
      `<div class=tile><div class=k><abbr title="Average GPU utilisation across all cards, as reported by nvidia-smi.">GPU util avg</abbr></div><div class=v class=mono>${gpuAvg==null?'--':gpuAvg+'<small>%</small>'}</div><div class=sub>${(s.gpu||[]).map((g,i)=>'GPU'+i+' '+((g.util==null?'?':g.util)+'%')).join(' &middot; ')||'no GPU data'}</div></div>`,
      `<div class=tile><div class=k><abbr title="Total requests since the gateway started, and current uptime.">Requests / uptime</abbr></div><div class=v class=mono>${s.total}</div><div class=sub>up ${uh}h${um}m &middot; peak lanes ${s.peak_inflight}</div></div>`,
    ].join('');
    // Routing mix
    const lp=s.local_pct,rp=s.remote_pct;
    $('#lrbar').innerHTML=`<i class=bl style=width:${lp}%></i><i class=br style=width:${rp}%></i>`;
    $('#lrtxt').innerHTML=`<span style=color:var(--grn)>&#9632;</span> local ${s.local} (${lp}%) &nbsp; <span style=color:var(--amb)>&#9632;</span> overflow ${s.remote} (${rp}%) &nbsp; peak lanes ${s.peak_inflight} &nbsp; uptime ${uh}h${um}m`;
    renderReasons(s.remote_reasons||{},s.remote||1);
    recentEvents=(s.events||[]);
    renderRecent();
    $('#tab_n_traffic').textContent=(s.remote||0)+' overflow';
  }

  // ---- WHY OVERFLOW: every reason gets a plain sentence + the live value of the setting
  // that causes it + a jump-to-setting link (items 11-20). Built from the exact strings
  // record_event() uses server-side -- see _route_completions() in keepalive-shim.py. ----
  const REASON_INFO={
    forced:      {why:'Full-remote mode is switched on -- every request goes straight to the paid provider.', field:'f_force_remote', label:v=>'force-remote = '+v, fix:'Switch the mode badge back to LOCAL FIRST.'},
    size:        {why:'Prompt + requested output was bigger than the single-request size cap.', field:'f_max_local_tokens', label:v=>'cap = '+Number(v).toLocaleString()+' tok', fix:'Raise the size cap in Settings if this box can actually hold it.'},
    'big-out':   {why:'Requested output tokens was at or above the big-output threshold.', field:'f_big_output', label:v=>'threshold = '+Number(v).toLocaleString()+' tok', fix:'Raise the big-output threshold if these should stay local.'},
    'big-prompt':{why:'The prompt was at or above the big-prompt threshold.', field:'f_big_prompt', label:v=>v>0?('threshold = '+Number(v).toLocaleString()+' tok'):'currently off (0)', fix:'Raise (or set) the big-prompt threshold.'},
    'local-down':{why:'The local engine was unhealthy when this request arrived.', field:'f_oom_backoff_secs', label:v=>'backoff = '+v+'s', fix:'Check the Local engine tile above and the engine logs.'},
    monster:     {why:'A huge prefill was already monopolizing the engine -- new arrivals overflow until it drains.', field:'f_monster_inflight', label:v=>'threshold = '+Number(v).toLocaleString()+' tok in flight', fix:'Raise the monster-inflight threshold, or expect this while a big job runs.'},
    'tiny-fast': {why:'This was a tiny/fast-lane request, but even the reserved tiny headroom was full.', field:'f_tiny_extra_lanes', label:v=>'extra tiny lanes = '+v, fix:'Add more tiny extra lanes.'},
    tokens:      {why:'A lane was free, but serving this would have exceeded the total in-flight context budget.', field:'f_token_budget', label:v=>'budget = '+Number(v).toLocaleString()+' tok', fix:'Raise the total in-flight context cap.'},
    prefill:     {why:'The engine already had about as much uncached prompt to prefill as the admission window allows (and this request was not small); it waited for a lane, then overflowed.', field:'f_prefill_admit_secs', label:v=>'window = '+v+' s of prefill', fix:'Raise the prefill admission window, or wait for the backlog to drain.'},
    'bg-yield':  {why:'Lanes existed, but they are reserved for interactive traffic -- this request was background.', field:'f_fg_reserved', label:v=>'reserved for interactive = '+v, fix:'Lower the reserved-for-interactive count, or accept background waits longer.'},
    cap:         {why:'All lanes were busy and this request waited past its queue timeout.', field:'f_local_budget', label:v=>'budget = '+v+' lane(s)', fix:'Raise Local budget (lanes), or raise the queue-wait timeout.'},
    failover:    {why:'Local accepted the request but errored or ran out of memory mid-flight, so it fell back to remote.', field:'f_oom_backoff_secs', label:v=>'backoff = '+v+'s', fix:'Check engine logs for the underlying crash.'},
    'force-remote':{why:'A deliberate full-remote maintenance window -- background traffic was rejected outright rather than billed.', field:'f_force_remote', label:v=>'force-remote = '+v, fix:'Turn off full-remote mode when the window ends.'},
  };
  function renderReasons(reasons,totalRemote){
    const entries=Object.entries(reasons).sort((a,b)=>b[1]-a[1]);
    if(!entries.length){$('#reasons').innerHTML='<span class=empty>No overflow yet -- everything has been served locally.</span>';return;}
    const max=entries[0][1];
    $('#reasons').innerHTML=entries.map(([key,n])=>{
      const info=REASON_INFO[key]||{why:'(undocumented reason: '+esc(key)+')',field:null,label:()=>'',fix:''};
      const val=info.field?CFG[info.field.slice(2)]:null;
      const pct=totalRemote?Math.round(100*n/totalRemote):0;
      const jump=info.field?`<a href="#" class=rz onclick="event.preventDefault();window.__gotoSetting('${info.field}')">${esc(info.field?info.label(val):'')} &rarr;</a>`:'';
      return '<div class=reasonrow><div><div class=rtitle>'+esc(key)+'</div><div class=rwhy>'+esc(info.why)+'</div>'+
        (info.fix?'<div class=rfix>&#128161; '+esc(info.fix)+'</div>':'')+
        '<div class=rbar><i style=width:'+Math.round(100*n/max)+'%></i></div></div>'+
        '<div class=rcount><b>'+n+'</b><div class=rz>'+pct+'% of overflow</div>'+(jump?'<div style="margin-top:3px">'+jump+'</div>':'')+'</div></div>';
    }).join('');
  }
  window.__gotoSetting=goToSetting;   // reasonrow links are built via innerHTML, so this needs a stable global

  function renderRecent(){
    const per=25,pg=paginate(recentEvents,'ev',per);
    $('#req_summary').textContent=pg.total? (pg.total+' events buffered (this process only) \u00b7 showing '+(pg.from+1)+'-'+pg.to) : 'No requests yet this uptime.';
    $('#ev').innerHTML=pg.slice.map(e=>`<tr><td class="rz mono">${fmtAgo(e.t)}</td><td>${esc(e.ep)}</td><td>${clientDisplay(e.client)}</td><td><span class="tag ${routeTag(e.d)}">${esc(e.d)}</span></td><td class=rz>${esc(e.r||'')}</td><td class=mono>${(e.ptok||0)}&rarr;${(e.maxtok||0)}${e.stream?' &#9889;':''}</td><td class=mono>${e.waited?e.waited+'s':''}</td></tr>`).join('')
      || '<tr><td colspan=7 class=empty>No requests logged yet this uptime.</td></tr>';
    wirePager('ev','#ev_prev','#ev_next','#ev_page',pg.pages,renderRecent);
  }

  // ==================== UNIFIED "NOW" LIST (items 1-10, 50-55) ====================
  // One list, one Kind column, one pager, one empty state -- instead of four hand-rolled
  // lists (live requests / agent lanes / frontier queue+windows / research) each with its
  // own pager and its own definition of "still running".
  const TERMINAL=new Set(['done','failed','error','degraded','cancelled']);
  let lanesData=null, winData=null, nFilter='';
  const N_STATE_LEGEND='<span class="sq run" style="vertical-align:-1px"></span> working &nbsp;'+
    '<span class="sq queued" style="vertical-align:-1px"></span> queued &nbsp;'+
    '<span class="sq stale" style="vertical-align:-1px"></span> stale (no heartbeat) &nbsp;'+
    '<span class="sq done" style="vertical-align:-1px"></span> done &nbsp;'+
    '<span class="sq blocked" style="vertical-align:-1px"></span> blocked';
  $('#n_state_legend').innerHTML=N_STATE_LEGEND;

  function buildUnified(){
    const rows=[];
    const L=lanesData||{};
    // requests (kind=request) -- in-flight HTTP calls, from /gateway/lanes .active
    (L.active||[]).forEach(a=>{
      const remote=a.route==='remote',queued=a.phase==='queued',held=a.phase==='held';
      const state=queued?'queued':held?'queued':remote?'stale':'run';
      // deadline estimate (item 7): how much of the configured wait is left before this
      // would overflow, computed client-side from the live config -- no backend change needed.
      let deadline=null;
      const guaranteedLocal=queued&&!a.bg&&CFG&&Number(CFG.interactive_never_overflow)===1;
      if(queued&&CFG&&!guaranteedLocal){
        const waitS=(a.bg?CFG.bg_wait_secs:CFG.local_wait_secs);
        if(waitS!=null)deadline=Math.max(0,Number(waitS)-(a.waited||0));
      }
      rows.push({kind:'request',who:clientDisplay(a.name,a.ip,a.ua),
        what:(a.preview?'"'+esc(a.preview)+'"':'<span class=rz>(no text preview)</span>')+(a.bg?' &middot; background':'')+(a.tiny?' &middot; tiny':'')+(a.model?' &middot; '+esc(a.model):''),
        state, badgeText: queued?'waiting for a lane'+(deadline!=null?' \u00b7 overflows in ~'+Math.round(deadline)+'s':(guaranteedLocal?' \u00b7 guaranteed local (never overflows)':'')):held?'holding for local (background) -- never billed':remote?'remote overflow \u00b7 '+esc(a.reason||''):a.phase==='local'?'local model'+(a.waited?' \u00b7 waited '+a.waited+'s':''):'routing',
        age:a.elapsed_s||0, meta:esc(a.ep)+(a.ptok?' \u00b7 '+Math.round(a.ptok/1000)+'K in':'')+(a.maxtok?' \u00b7 '+a.maxtok+' max out':'')+(a.stream?' \u00b7 \u26a1 stream':''),
        sortkey:(a.name||'')+' '+(a.preview||'')});
    });
    // agent lanes (kind=lane) -- long-running background workers (STATUS-file heartbeat)
    (L.lanes||[]).forEach(l=>{
      const st=l.status||'';const m=st.match(/^(RUNNING|DONE|BLOCKED)\s*\|\s*([^|]*)\|\s*(.*)$/);
      const stg=m?m[1]:(st?'RUNNING':'UNKNOWN');const text=m?m[3].trim():(st||('newest file: '+l.newest));
      const stale=stg==='RUNNING'&&l.age_s>900;
      const state=stg==='DONE'?'done':stg==='BLOCKED'?'blocked':stale?'stale':stg==='RUNNING'?'run':'empty';
      rows.push({kind:'lane',who:'<span class=role>'+esc(l.lane)+'</span>',what:esc(text),state,
        badgeText: state==='done'?'done':state==='blocked'?'blocked':state==='stale'?'no heartbeat for '+dur(l.age_s):state==='run'?'working':'idle',
        age:l.age_s||0, meta:esc((l.root||'').replace(/^\/home\/kevin\//,'~/')), sortkey:l.lane||''});
    });
    // research jobs (kind=research)
    (L.research||[]).forEach(j=>{
      const running=!TERMINAL.has(j.status),zero=/^0\//.test(j.claims||'');
      const state=running?'run':(j.status==='failed'||j.status==='error')?'blocked':(j.status==='degraded'||zero)?'stale':'done';
      rows.push({kind:'research',who:'<span class=rawid>'+esc((j.id||'').slice(-6))+'</span>',what:esc(j.q)+' &middot; '+esc(j.depth)+(j.agents?' \u00b7 '+esc(j.agents)+' agents':'')+(j.tokens?' \u00b7 '+Math.round(j.tokens/1000)+'K tok':''),state,
        badgeText: running?esc(j.phase||j.status)+(j.progress?' \u00b7 '+esc(j.progress):''):j.status==='degraded'?'degraded -- no usable evidence':zero?'0 claims survived':j.claims?esc(j.claims)+' claims':'',
        age:j.elapsed_s||0, meta:'<a href="/gateway/research/'+encodeURIComponent(j.id||'')+'" target=_blank rel=noopener>report &rarr;</a>', sortkey:j.q||'',
        href:'/gateway/research/'+encodeURIComponent(j.id||'')});
    });
    // frontier windows (kind=window) -- this now covers what used to be a SEPARATE "frontier
    // queue" list too (item 51): queued/running/done are all just window states.
    ((winData&&winData.windows)||[]).forEach(w=>{
      const state=w.state==='queued'?'queued':w.failed?'blocked':w.state==='running'?'run':'done';
      rows.push({kind:'window',who:'<span class=role>'+esc(w.name)+'</span>',what:esc(w.description||'(no header comment found)'),state,
        badgeText: w.state==='queued'?'queued':w.duration_s!=null?dur(w.duration_s)+(w.state==='running'?' so far':''):w.state,
        age:w.duration_s||0, meta:w.log_url?'<a href="'+w.log_url+'" target=_blank rel=noopener>log &rarr;</a>':'', sortkey:w.name||''});
    });
    return rows;
  }
  function kindLabel(k){return {request:'request',lane:'agent lane',window:'window',research:'research'}[k]||k;}
  function renderNow(){
    const all=buildUnified();
    const summary={request:0,lane:0,window:0,research:0};
    let longest=null;
    all.forEach(r=>{if(r.state==='run'){summary[r.kind]=(summary[r.kind]||0)+1;if(!longest||r.age>longest.age)longest=r;}});
    $('#now_summary').innerHTML=
      '<span><b>'+summary.request+'</b><span class=lbl>requests in flight</span></span>'+
      '<span><b>'+summary.lane+'</b><span class=lbl>agent lanes working</span></span>'+
      '<span><b>'+summary.window+'</b><span class=lbl>windows running</span></span>'+
      '<span><b>'+summary.research+'</b><span class=lbl>research running</span></span>';
    const longestEl=$('#longest');
    if(longest&&longest.age>180){longestEl.hidden=false;longestEl.innerHTML='\u23f1 longest still running: <b>'+kindLabel(longest.kind)+'</b> '+longest.who+' &mdash; '+dur(longest.age);}
    else longestEl.hidden=true;
    $('#tab_n_now').textContent=(summary.request+summary.lane+summary.window+summary.research)+' active';

    const showDone=$('#n_showdone').checked;
    let rows=all.filter(r=>(!nFilter||r.kind===nFilter)&&(showDone||!TERMINAL.has(r.state)&&r.state!=='done'));
    const s=SORT.now;
    if(s.key==='kind')rows=sortArr(rows,'kind',s.dir);
    else if(s.key==='who')rows=sortArr(rows,'sortkey',s.dir);
    else if(s.key==='age')rows=sortArr(rows,'age',s.dir);
    else rows=rows.slice().sort((a,b)=>(b.state==='run')-(a.state==='run')||b.age-a.age); // default: working-first, then oldest

    const per=25,pg=paginate(rows,'now',per);
    $('#n_rows').innerHTML=pg.slice.map(r=>{
      const inner='<span class="sq '+r.state+'"></span>';
      const body='<div class=tname>'+r.who+'</div><div class=tstat>'+r.what+'</div>';
      const badge='<span class="badge '+(r.state==='run'?'run':r.state==='done'?'ok':r.state==='blocked'?'err':'warn')+'">'+esc(r.badgeText)+'</span>';
      return '<tr><td>'+inner+'</td><td><span class="badge kind">'+kindLabel(r.kind)+'</span></td><td>'+body+'</td><td>'+badge+'</td><td class=mono>'+dur(r.age)+'</td><td class=tmeta>'+(r.meta||'')+'</td></tr>';
    }).join('')||('<tr><td colspan=6 class=empty>'+(all.length?'Nothing matches this filter right now.':'Nothing in flight -- gateway is idle. That is normal, not a problem.')+'</td></tr>');
    wirePager('now','#n_prev','#n_next','#n_page',pg.pages,renderNow);
    $('#n_err').textContent=[(lanesData&&lanesData.errors)||[],(winData&&winData.errors)||[]].flat().join(', ');
    // health checks (estate watchdog -- kept, but visually a separate box now, not the top of the page)
    const H=(lanesData&&lanesData.health)||{},HC=H.checks||[];
    $('#n_health').textContent=H.present?HC.length:'-';
    $('#health_line').innerHTML=H.present?('overall <b style="color:'+(H.overall==='ok'?'var(--grn)':H.overall==='warn'?'var(--amb)':'var(--red)')+'">'+esc(H.overall||'?')+'</b> \u00b7 last run '+(H.age_s!=null?ago(H.age_s):'?')+(H.paused?' \u00b7 <span class="badge warn">PAUSED</span>':'')):'estate watchdog not installed yet';
    $('#t_health').innerHTML=HC.map(c=>'<span title="'+esc(c.detail||'')+'"><span class="sq '+(c.status==='OK'?'done':c.status==='WARN'?'stale':c.status==='CRIT'?'blocked':'empty')+'" style="margin:0 5px 0 0;vertical-align:-1px"></span>'+esc(c.name)+' <span class=rz>'+esc((c.detail||'').slice(0,48))+'</span></span>').join('')||'<span class=empty>'+(H.present?'no checks reported':'no watchdog state yet')+'</span>';
    const ha=$('#health_alert');if(H.alert){ha.hidden=false;ha.textContent='ALERT \u00b7 '+H.alert.slice(0,400);}else{ha.hidden=true;}
  }
  $$('.chip[data-k]').forEach(c=>c.addEventListener('click',()=>{
    nFilter=c.dataset.k;PAGE.now=1;
    $$('.chip[data-k]').forEach(x=>x.classList.toggle('on',x===c));
    renderNow();
  }));
  $('#k_all').classList.add('on');
  $('#n_showdone').addEventListener('change',()=>{PAGE.now=1;renderNow();});
  wireSort('#n_table','now',renderNow);
  $('#n_csv').addEventListener('click',()=>{
    const rows=buildUnified().filter(r=>!nFilter||r.kind===nFilter);
    csvDownload('gateway-now.csv',[['kind','who','status','age_s'],...rows.map(r=>[r.kind,r.who.replace(/<[^>]+>/g,''),r.badgeText,Math.round(r.age)])]);
  });

  let busyLanes=false;
  async function tickLanes(){if(busyLanes)return;busyLanes=true;
    try{lanesData=await(await fetch('/gateway/lanes',{cache:'no-store'})).json();lastOk=Date.now();renderNow();}
    catch(e){setBanner('lanes','background-task feed unreachable: '+e,'err');}
    finally{busyLanes=false;}
  }
  let busyWin=false;
  async function tickWindows(){if(busyWin)return;busyWin=true;
    try{winData=await(await fetch('/gateway/windows',{cache:'no-store'})).json();renderNow();}
    catch(e){setBanner('windows','windows feed unreachable: '+e,'err');}
    finally{busyWin=false;}
  }

  // ==================== TELEMETRY (uses /gateway/telemetry) ====================
  function sparkline(id,series){
    const svg=document.getElementById(id);if(!svg)return;
    const W=300,H=56,pad=3;
    const lines=(series||[]).filter(s=>s&&s.length);
    if(!lines.length){svg.innerHTML='';return;}
    const n=Math.max(...lines.map(s=>s.length));
    const all=[].concat(...lines).filter(v=>v!=null&&isFinite(v));
    let lo=all.length?Math.min(...all):0, hi=all.length?Math.max(...all):1;
    lo=Math.min(0,lo);if(hi<=lo)hi=lo+1;
    const x=i=>pad+(W-2*pad)*(n<=1?0:i/(n-1));
    const y=v=>H-pad-(H-2*pad)*((v-lo)/(hi-lo));
    const cls=['s1','s2','s3'];
    svg.innerHTML=lines.map((s,li)=>{
      const pts=s.map((v,i)=>(v==null||!isFinite(v))?null:x(i).toFixed(1)+','+y(v).toFixed(1)).filter(Boolean).join(' ');
      return pts?'<polyline class="'+cls[li%3]+'" points="'+pts+'"></polyline>':'';
    }).join('');
    return {n,lo,hi};
  }
  // sample cadence is fixed server-side at 2s (TELEM_SAMPLE_SECS) -- used only to label the
  // window ("last ~Nm"), never to compute anything routing-relevant.
  const TELEM_SAMPLE_SECS=2;
  function sparkFoot(id,n,cur){
    const el=document.getElementById(id);if(!el)return;
    const win=n?dur(n*TELEM_SAMPLE_SECS):'--';
    el.textContent='last ~'+win+(cur!=null?' \u00b7 now '+cur:'');
  }
  function seriesOf(fast,path){return fast.map(s=>{let v=s;for(const k of path){v=v&&v[k];if(v==null)return null;}return v;});}
  let telemData=null,perClientList=[],errorFeed=[],pcWindow='uptime';
  function renderTelem(){
    if(!telemData)return;
    const t=telemData;
    const fast=(t.series&&t.series.fast)||[];
    const eng=(t.latest&&t.latest.engine)||{};
    const tpsNow=$('#tps_now'),ttftLine=$('#ttft_line');
    if(tpsNow){tpsNow.innerHTML=eng.ok&&(eng.prompt_tok_s!=null||eng.gen_tok_s!=null)?(((eng.prompt_tok_s||0)+(eng.gen_tok_s||0)).toFixed(0)):'--';}
    if(ttftLine){ttftLine.innerHTML=eng.ok&&eng.ttft_p50!=null?(eng.ttft_p50.toFixed(2)+'s / '+(eng.ttft_p95||0).toFixed(2)+'s'):'--';}
    $('#telem_engok').innerHTML=eng.ok?'<span class="dot up"></span> scraping OK':
      ('<span class="dot down"></span> '+(eng.age_s!=null?('stale '+Math.round(eng.age_s)+'s ago ('+esc(eng.err||'')+')'):('never scraped yet ('+esc(eng.err||'engine down')+')')));
    setBanner('engine', eng.ok?null:'Engine Prometheus scrape is failing -- telemetry sparklines are stale.', 'warn');
    $('#telem_tiles').innerHTML=[
      tile('Tokens/s now',eng.ok&&(eng.prompt_tok_s!=null||eng.gen_tok_s!=null)?(((eng.prompt_tok_s||0)+(eng.gen_tok_s||0)).toFixed(0)):'--'),
      tile('TTFT (time to first token) p50 <small>/ p95</small>',eng.ok&&eng.ttft_p50!=null?(eng.ttft_p50.toFixed(2)+'s <small>/ '+(eng.ttft_p95||0).toFixed(2)+'s</small>'):'--'),
      tile('KV-cache % (attention-cache memory used)',eng.ok&&eng.kv_cache_pct!=null?(eng.kv_cache_pct.toFixed(0)+'<small>%</small>'):'--'),
      tile('Engine running <small>/ waiting</small>',eng.ok?((eng.running==null?'--':eng.running)+' <small>/ '+(eng.waiting==null?'--':eng.waiting)+'</small>'):'--'),
    ].join('');
    let m;
    m=sparkline('sp_toks',[seriesOf(fast,['engine','prompt_tok_s']),seriesOf(fast,['engine','gen_tok_s'])]);sparkFoot('sf_toks',fast.length,eng.ok?((eng.prompt_tok_s||0).toFixed(0)+'p/'+(eng.gen_tok_s||0).toFixed(0)+'g tok/s'):null);
    m=sparkline('sp_ttft',[seriesOf(fast,['engine','ttft_p50']),seriesOf(fast,['engine','ttft_p95'])]);sparkFoot('sf_ttft',fast.length,eng.ttft_p50!=null?eng.ttft_p50.toFixed(2)+'s':null);
    m=sparkline('sp_tpot',[seriesOf(fast,['engine','tpot_p50']),seriesOf(fast,['engine','tpot_p95'])]);sparkFoot('sf_tpot',fast.length,eng.tpot_p50!=null?eng.tpot_p50.toFixed(2)+'s':null);
    m=sparkline('sp_kv',[seriesOf(fast,['engine','kv_cache_pct']),seriesOf(fast,['gateway','inflight'])]);sparkFoot('sf_kv',fast.length,eng.kv_cache_pct!=null?eng.kv_cache_pct.toFixed(0)+'%':null);
    const pw=(gi)=>fast.map(s=>{const g=s.gpu&&s.gpu[gi];return(g&&g.power_w!=null&&g.power_limit_w)?100*g.power_w/g.power_limit_w:null;});
    m=sparkline('sp_gpu0',[seriesOf(fast,['gpu',0,'util']),seriesOf(fast,['gpu',0,'temp_c']),pw(0)]);sparkFoot('sf_gpu0',fast.length,null);
    m=sparkline('sp_gpu1',[seriesOf(fast,['gpu',1,'util']),seriesOf(fast,['gpu',1,'temp_c']),pw(1)]);sparkFoot('sf_gpu1',fast.length,null);
    m=sparkline('sp_remote',[seriesOf(fast,['gateway','remote_share_pct'])]);sparkFoot('sf_remote',fast.length,null);
    const pc=t.per_client||{};
    perClientList=Object.keys(pc).sort((a,b)=>pc[b].requests-pc[a].requests).map(n=>({n:n,c:pc[n]}));
    renderPerClient();
    errorFeed=t.errors||[];
    renderErrorFeed();
  }
  function renderPerClient(){
    const per=25,pg=paginate(perClientList,'pc',per);
    $('#pc_filtered_note').textContent='uptime-scoped rollup (switch the window selector for a 24h view backed by the on-disk log)';
    $('#telem_clients').innerHTML=pg.slice.map(x=>{const n=x.n,c=x.c;
      return '<tr><td><a href="#" onclick="event.preventDefault();window.__filterHistory(\''+esc(n).replace(/'/g,"\\'")+'\')">'+clientDisplay(n)+'</a></td><td class=mono>'+c.requests+'</td><td class=mono>'+c.local+'</td><td class=mono>'+c.remote+
        '</td><td class=mono>'+c.tokens_out+(c.tokens_out_lb>0?' <span class=rz title="includes chunk-count lower bounds">(~)</span>':'')+
        '</td><td class=mono>'+(c.wait_avg_s!=null?c.wait_avg_s+'s':'--')+'</td><td class=mono>'+(c.ttft_avg_s!=null?c.ttft_avg_s+'s':'--')+
        '</td><td class=mono>'+c.errors+'</td><td class=mono>'+(c.cost_est_usd?'$'+c.cost_est_usd.toFixed(4):'--')+'</td></tr>';
    }).join('')||'<tr><td colspan=9 class=empty>No completed requests yet this uptime.</td></tr>';
    wirePager('pc','#pc_prev','#pc_next','#pc_page',pg.pages,renderPerClient);
  }
  window.__filterHistory=function(name){$('#h_client').value=name;hPage=1;tickHistory();applyTab('traffic',true);$('#h_client').scrollIntoView({block:'center'});};
  $('#pc_window').addEventListener('change',async e=>{
    pcWindow=e.target.value;
    if(pcWindow==='uptime'){renderPerClient();return;}
    try{const s=await(await fetch('/gateway/history/summary?hours=24',{cache:'no-store'})).json();
      const pc=s.per_client||{};
      perClientList=Object.keys(pc).sort((a,b)=>pc[b].requests-pc[a].requests).map(n=>({n:n,c:{requests:pc[n].requests,local:pc[n].local,remote:pc[n].remote,tokens_out:pc[n].tokens_out,errors:pc[n].errors,wait_avg_s:null,ttft_avg_s:null,cost_est_usd:null}}));
      $('#pc_filtered_note').textContent='last 24h, from the on-disk log';
      renderPerClient();
    }catch(e2){setBanner('history','24h summary unreachable: '+e2,'err');}
  });
  function renderErrorFeed(){
    const per=25,pg=paginate(errorFeed,'ef',per);
    $('#telem_errors').innerHTML=pg.slice.map(e=>'<tr><td class="rz mono">'+fmtAgo(e.t)+'</td><td>'+clientDisplay(e.client)+'</td><td>'+esc(e.ep)+
      '</td><td><span class="tag '+routeTag(e.route)+'">'+esc(e.route)+'</span></td><td class=rz>'+esc(e.reason)+
      '</td><td class=mono>'+(e.status||'')+'</td></tr>').join('')||'<tr><td colspan=6 class=empty>No errors recorded this uptime.</td></tr>';
    wirePager('ef','#ef_prev','#ef_next','#ef_page',pg.pages,renderErrorFeed);
  }
  async function tickTelemetry(){
    try{telemData=await(await fetch('/gateway/telemetry',{cache:'no-store'})).json();renderTelem();}
    catch(e){setBanner('telemetry','telemetry feed unreachable: '+e,'err');}
  }

  // ==================== HISTORY (server-side paginated /gateway/history) ====================
  let hPage=1,hTimer=null;
  function hLimit(){return parseInt($('#h_limit').value)||25;}
  async function tickHistory(){
    const cq=($('#h_client').value||'').trim(),rq=($('#h_route').value||'').trim();
    const limit=hLimit();
    const u='/gateway/history?limit='+limit+'&page='+hPage+(cq?'&client='+encodeURIComponent(cq):'')+(rq?'&route='+encodeURIComponent(rq):'');
    let h;try{h=await(await fetch(u,{cache:'no-store'})).json();}
    catch(e){$('#h_rows').innerHTML='<tr><td colspan=8 class=empty>history feed unreachable: '+esc(e)+'</td></tr>';return;}
    const items=h.items||[];
    $('#h_rows').innerHTML=items.map(r=>{
      const badge='<span class="tag '+routeTag(r.route)+'">'+esc(r.route||'?')+'</span>'+(r.reason?' <span class=rz>'+esc(r.reason)+'</span>':'');
      const out=r.outtok!=null?r.outtok:(r.outtok_lb!=null?'~'+r.outtok_lb:0);
      return '<tr><td class="rz mono">'+fmtAgo(r.t)+'</td><td>'+clientDisplay(r.client)+'</td><td>'+badge+
        '</td><td class=rz title="'+esc(r.preview||'')+'">'+esc((r.preview||'').slice(0,60))+
        '</td><td class=mono>'+(r.ptok||0)+'&rarr;'+out+
        '</td><td class=mono>'+(r.ttft!=null?r.ttft.toFixed(2)+'s':'--')+
        '</td><td class=mono>'+fmtDur(r.duration)+'</td><td class=mono>'+(r.status||'')+'</td></tr>';
    }).join('')||('<tr><td colspan=8 class=empty>No matching requests logged'+((cq||rq)?' for this filter.':' yet.')+'</td></tr>');
    const lg=h.log||{};
    $('#h_logstate').textContent='logged '+(lg.written||0)+' since restart'+(lg.dropped_cap?' \u00b7 '+lg.dropped_cap+' dropped (daily cap)':'')+(lg.dropped_queue?' \u00b7 '+lg.dropped_queue+' dropped (queue full)':'')+(lg.capped_today?' \u00b7 TODAY LOG AT SIZE CAP':'');
    $('#h_page').textContent=hPage;
    $('#h_prev').disabled=hPage<=1;
    $('#h_next').disabled=!h.has_more;
  }
  $('#h_client').addEventListener('input',()=>{clearTimeout(hTimer);hTimer=setTimeout(()=>{hPage=1;tickHistory();},300);});
  $('#h_route').addEventListener('input',()=>{clearTimeout(hTimer);hTimer=setTimeout(()=>{hPage=1;tickHistory();},300);});
  $('#h_limit').addEventListener('change',()=>{hPage=1;tickHistory();});
  $('#h_prev').addEventListener('click',()=>{if(hPage>1){hPage--;tickHistory();}});
  $('#h_next').addEventListener('click',()=>{hPage++;tickHistory();});

  // ==================== LOCAL MODELS + CONFIG ====================
  async function loadLocalModels(){
    try{const d=await(await fetch('/gateway/models/local')).json();
      $('#mdir').textContent=d.models_dir||'';
      $('#lm').innerHTML=(d.models||[]).map(m=>{
        const badge=m.live?'<span class="tag local">LIVE</span>':'';
        const btn=m.servable&&!m.live?`<button class="ghost lmsw" data-p="${esc(m.path)}" style="font-size:11px;padding:3px 10px">Switch</button>`:
                  (m.servable?'<span class=rz>current</span>':'<span class=rz style=color:var(--amb)>not servable</span>');
        return `<tr><td>${badge}</td><td class=mono>${esc(m.name)}</td><td class=mono>${esc(m.gb)} GB</td><td class=rz>${esc(m.desc)}</td><td>${btn}</td></tr>`;
      }).join('')||'<tr><td colspan=5 class=rz>no checkpoints found</td></tr>';
      $$('.lmsw').forEach(b=>b.addEventListener('click',()=>switchLocal(b.dataset.p)));
    }catch(e){}
  }
  async function switchLocal(p){
    if(!confirm('Switch the local engine to:\n\n'+p+'\n\nThe engine restarts. Traffic falls back to remote overflow until it is healthy.'))return;
    $('#lmmsg').textContent='switching... engine restarting';
    try{const r=await adminFetch('/gateway/models/local',{method:'POST',body:JSON.stringify({path:p})});
      const d=await r.json();
      $('#lmmsg').textContent=d.error?('X '+d.error):('OK switching to '+d.switching_to+' ('+d.quant+') -- '+d.note);
      setTimeout(loadLocalModels,15000);
    }catch(e){$('#lmmsg').textContent='X '+e;}
  }
  async function loadCfg(){
    try{const c=await(await fetch('/gateway/config')).json();
      CFG=c;
      FR=c.force_remote?1:0;
      MODE=c.mode||(c.local_only?'full_local':(c.force_remote?'full_remote':'local_first'));renderMode();
      for(const [k,v] of Object.entries(c)){const el=document.getElementById('f_'+k);if(el&&el.type!=='password')el.value=v;}
      $('#keystate').textContent=c.remote_key_display?('current: '+c.remote_key_display):'not set';
      const ab=$('#authbadge'),at=$('#authtext');
      if(c.admin_token_set){ab.textContent='admin-gated';ab.className='authbadge on';
        at.innerHTML='An admin token IS set. Saving settings or switching the local model from this page requires it -- you will be prompted once per browser and it is then remembered in this browser only (localStorage).';}
      else{ab.textContent='\u26a0 open (no admin token)';ab.className='authbadge off';
        at.innerHTML='<b style=color:var(--amb)>No admin token is set.</b> Anyone on the LAN who can reach this page can change settings or switch the local model with no login. Set <span class=mono>SHIM_ADMIN_TOKEN_FILE</span> (or <span class=mono>SHIM_ADMIN_TOKEN</span>) and restart the service to close this.';}
    }catch(e){}
  }
  async function saveCfg(){
    const b={};
    $$('input[id^=f_]').forEach(el=>{const k=el.id.slice(2);
      if(el.type==='password'){if(el.value.trim())b[k]=el.value.trim();}
      else if(el.value!=='')b[k]=el.value;
    });
    $('#savemsg').textContent='saving...';
    try{const r=await adminFetch('/gateway/config',{method:'POST',body:JSON.stringify(b)});
      const d=await r.json();$('#savemsg').textContent='OK saved: '+(d.changed||[]).join(', ');$('#f_remote_key').value='';loadCfg();}
    catch(e){$('#savemsg').textContent='X '+e;}
  }
  $('#save').addEventListener('click',saveCfg);
  $('#f_preset').addEventListener('change',e=>{const v=e.target.value;if(!v)return;const [b,m]=v.split('|');$('#f_remote_base').value=b;$('#f_remote_model').value=m;});

  // ==================== SHARED CLOCK (item 56): one setInterval instead of five, and
  // everything pauses while the tab is hidden instead of repainting invisibly forever. ====================
  const TASKS=[
    {every:1500, fn:tickStats, due:0},
    {every:2000, fn:tickLanes, due:0},
    {every:10000, fn:tickWindows, due:0},
    {every:2000, fn:tickTelemetry, due:0},
    {every:10000, fn:tickHistory, due:0},
    {every:15000, fn:loadModels, due:0},
  ];
  function clockTick(){
    if(document.hidden)return;
    const now=Date.now();
    TASKS.forEach(t=>{if(now>=t.due){t.due=now+t.every;t.fn();}});
  }
  setInterval(clockTick,500);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden){TASKS.forEach(t=>t.due=0);clockTick();}});

  // ==================== BOOT ====================
  loadModels();loadCfg();loadLocalModels();
  tickStats();tickLanes();tickWindows();tickTelemetry();tickHistory();
})();
</script></body></html>"""

ALIASES_HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Gateway aliases</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--dim:#8b949e;--blu:#58a6ff;--red:#f85149;--grn:#3fb950}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}.wrap{max-width:1000px;margin:auto;padding:20px}.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin:14px 0}h1{font-size:20px}h2{font-size:15px;border-bottom:1px solid var(--bd);padding-bottom:8px}label{display:flex;flex-direction:column;gap:4px;color:var(--dim);font-size:12px}input{background:var(--bg);border:1px solid var(--bd);border-radius:6px;color:var(--fg);padding:8px;font:inherit}form .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.wide{grid-column:1/-1}button{border:1px solid var(--bd);border-radius:6px;background:var(--blu);color:#06111f;padding:8px 14px;font-weight:600;cursor:pointer}.danger{background:transparent;color:var(--red)}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid var(--bd);padding:8px}small,.muted{color:var(--dim)}#msg{margin:10px 0}.ok{color:var(--grn)}.err{color:var(--red)}
</style></head><body><div class=wrap><a href=/gateway/dashboard>&larr; gateway dashboard</a><h1>Gateway provider aliases</h1>
<p class=muted>Use <code>estate</code> for the dashboard mode, <code>estate-local</code> for local-only, and <code>estate-remote</code> for the default remote. Custom aliases always target their configured OpenAI-compatible endpoint. Context limits are enforced per endpoint.</p>
<div class=card><h2>Admin token</h2><label>Token (stored only in this browser)<input id=token type=password autocomplete=off></label></div>
<div class=card><h2>Create or update custom alias</h2><form id=form><div class=grid>
<label>Alias name <small>e.g. estate-openai</small><input id=name required pattern="[A-Za-z][A-Za-z0-9._-]{1,63}"></label>
<label>Model id<input id=model required></label><label class=wide>Base URL<input id=base type=url placeholder=https://api.openai.com/v1 required></label>
<label>API key <small>blank on update keeps existing key</small><input id=key type=password autocomplete=off></label>
<label>Context limit (tokens)<input id=context_limit type=number min=1024 value=128000 required></label>
<label>Maximum output (tokens)<input id=max_output type=number min=1 value=16384 required></label>
<label><span>Enabled</span><input id=enabled type=checkbox checked></label>
</div><p><button type=submit>Save alias</button></p></form><div id=msg></div></div>
<div class=card><h2>Configured aliases</h2><div id=list>Loading...</div></div>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const headers=()=>{const t=$('token').value.trim();return t?{'X-Admin-Token':t,'Content-Type':'application/json'}:{'Content-Type':'application/json'}};
if(localStorage.gatewayAdminToken){$('token').value=localStorage.gatewayAdminToken} $('token').onchange=()=>localStorage.gatewayAdminToken=$('token').value;
async function load(){const r=await fetch('/gateway/aliases',{cache:'no-store'}),d=await r.json();let h='<p class=muted>Built-ins: '+d.builtins.map(esc).join(', ')+'</p>';
if(!d.aliases.length)h+='<p class=muted>No custom aliases.</p>';else h+='<table><thead><tr><th>alias</th><th>endpoint</th><th>model</th><th>context</th><th>status</th><th></th></tr></thead><tbody>'+d.aliases.map(a=>'<tr><td><code>'+esc(a.name)+'</code></td><td>'+esc(a.base)+'</td><td>'+esc(a.model)+'</td><td>'+Number(a.context_limit).toLocaleString()+' / '+Number(a.max_output).toLocaleString()+'</td><td>'+ (a.enabled?'enabled':'disabled')+'</td><td><button class=danger data-del="'+esc(a.name)+'">Delete</button></td></tr>').join('')+'</tbody></table>';$('list').innerHTML=h;document.querySelectorAll('[data-del]').forEach(b=>b.onclick=async()=>{if(!confirm('Delete '+b.dataset.del+'?'))return;const r=await fetch('/gateway/aliases',{method:'DELETE',headers:headers(),body:JSON.stringify({name:b.dataset.del})});$('msg').textContent=(await r.json()).error||'Deleted';load()})}
$('form').onsubmit=async e=>{e.preventDefault();const d={name:$('name').value,base:$('base').value,model:$('model').value,context_limit:Number($('context_limit').value),max_output:Number($('max_output').value),enabled:$('enabled').checked};if($('key').value)d.key=$('key').value;const r=await fetch('/gateway/aliases',{method:'POST',headers:headers(),body:JSON.stringify(d)}),j=await r.json();$('msg').className=r.ok?'ok':'err';$('msg').textContent=j.error||('Saved '+j.saved);if(r.ok){$('key').value='';load()}};load();
</script></div></body></html>"""
