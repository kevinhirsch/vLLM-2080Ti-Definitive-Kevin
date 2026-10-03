# Gateway dashboard audit

Scope: every number, chart and table on the previous `/gateway/dashboard` page (the 1,145-line inline `DASHBOARD_HTML` in
`keepalive-shim.py`), checked against the live endpoints and the engine's `/metrics` on 2026-10-02 (gateway process up
about 2.6 h, engine on :8001, a planned offline window open part of the time).

How each row was checked: fetch the endpoint, apply the page's own JS transform by hand, compare with an independent
source (engine `/metrics`, the spend ledger, the on-disk request log, the shim source). Units are the units the page
displayed or implied.

**Verdicts**

* `OK` the value is what its label says.
* `WRONG` mislabelled, misleading, or wrongly computed: the number is real but does not mean what the label says.
* `STALE` shown without the caveat that it is a configured/old/empty value, or an explanation list that no longer covers the live data.

The new page (`deploy/bin/gateway_dashboard.html`) keeps every row below (see the last column) and adds the data listed in
section 3. Every number on it carries a source tag: LIVE, MEASURED (with its window), CONFIGURED, or ESTIMATE.

## 1. Inventory of the previous page

| # | Item (as labelled) | Source | Transform | Units | Verdict | Finding | Now |
|---|---|---|---|---|---|---|---|
| 1 | Status dot | client clock | green under 8 s since last stats **or lanes** success | seconds | OK | a failing `/gateway/stats` was masked by a succeeding `/gateway/lanes` | header verdict + "updated N s ago" (stats and capacity only) + per-feed freshness table |
| 2 | Mode badge ("LOCAL-FIRST") | `/gateway/config.mode` | label lookup | enum | WRONG | shows the *configured* policy as if it were what is happening; during a planned offline window routing is remote-only while the badge says LOCAL-FIRST | Summary "Where new requests go" uses `/gateway/capacity.mode`; configured policy shown separately, tagged CONFIGURED |
| 3 | Mode buttons | `POST /gateway/config` | sets `local_only` / `force_remote` | enum | OK | | Routing section |
| 4 | Admin badge | `config.admin_token_set` | boolean | | OK | | Settings |
| 5 | "updated Ns ago" | client | now - lastOk | s | OK | | header |
| 6 | Overflow provider | `stats.remote_model` | | text | OK | | Health |
| 7 | Local engine up/DOWN | `stats.local_healthy` | cached health probe | boolean | OK | cached for the health TTL, so up to a few seconds old | Summary + Health, LIVE |
| 8 | Model name | `/v1/models.data[0].id` | strip path, cut at 26 chars | text | OK | | Summary + Health |
| 9 | Lanes `a/b` | `stats.inflight / stats.budget` | | capacity units | OK | units, not requests (a large request costs several); engine can be busy with work that never passed the gateway (observed: engine 12 running, lanes 0) | Engine tile now says so |
| 10 | "+N waiting (x interactive, y bg)" | `stats.waiting`, `waiting_by_class` | | requests | OK | only the old 2-way split; four work classes were not shown | Work classes section |
| 11 | "context reserved NK / NK cap" | `stats.inflight_reserved_tokens` / `stats.token_budget` | divide by 1000 | tokens | STALE | the cap is the configured 500,000, a number frozen against an engine that no longer exists; real KV pool is 922,358 tokens | Capacity: token budget with its source (live-derived / override / configured), pool live vs configured with a stale flag |
| 12 | OOM backoff | `stats.backoff` | | s | OK | | banner + Health |
| 13 | "Served local %" | `stats.local_pct` | `local / total` | % | WRONG | lifetime counters over 51 days with no window shown (53.9%) while the last 15 min was 91-98% remote | Routing: 15 min / 1 h / 24 h / all time, each labelled |
| 14 | "overflow %" | `stats.remote_pct` | `remote / total` | % | WRONG | same; also local% + overflow% is 91.3%, the rest (held, rejected) was silently omitted | Routing mix shows local, paid, held, rejected |
| 15 | "avg wait Ns" (33.5) | `stats.avg_wait` | `waited_total / waited_n` | s | WRONG | mean over only the requests that waited at all, lifetime; 24 h log says admission wait p50 = 0 s, p90 = 0 s | Work classes: per-class wait typical/slow (15 min); latency by class (24 h) |
| 16 | "Tokens/s now" | `telemetry.latest.engine.prompt_tok_s + gen_tok_s` | **sum** | tok/s | WRONG | adds prompt tokens (including ones served from cache) to generated tokens, over a 2 s window; example 1-minute sample: 5,131 prompt + 88 generated = 5,219 "tok/s" | Throughput: decode and prompt shown separately, with their windows |
| 17 | "time to first token p50 / p95" | `latest.engine.ttft_p50/p95` | 2 s window quantile | s | STALE | the engine reports a windowed quantile only if a request finished inside the 2 s window: non-null in 20% of samples, so the tile was blank 80% of the time | three views: last 5 min, since engine start, last 24 h |
| 18 | GPU util avg | mean of `stats.gpu[].util` | | % | OK | | Summary + GPU table |
| 19 | GPU per-card util in sub-line | `stats.gpu[].util` | | % | OK | | GPU table |
| 20 | Requests total | `stats.total` | | requests | OK | lifetime, unlabelled | labelled "all time" |
| 21 | "up 1235h37m" | `stats.uptime` | h, m | h | WRONG | `uptime` is the age of the *persisted counters* (51 days), not of the process, which had been up 2.6 h | `stats.process_uptime` (new field) and `stats.stats_since` |
| 22 | Peak lanes | `stats.peak_inflight` | | lanes | OK | lifetime | Capacity |
| 23 | Routing mix bar | `stats.local_pct / remote_pct` | | % | WRONG | as rows 13-14 | Routing |
| 24 | Reason list: counts | `stats.remote_reasons` | sort desc | requests | OK | sums exactly to `stats.remote` (110,250) | Routing, window selectable |
| 25 | Reason list: "% of overflow" | counts / `stats.remote` | | % | OK | | |
| 26 | Reason explanations | static `REASON_INFO` | | text | STALE | 6 of the 18 live reasons had no explanation ("undocumented"): `perf`, `alias`, `local-offline`, `predicted`, `intent`, `full-local-remote-alias`: 43.8% of all overflow, including the largest, `perf` (34,790) | all 21 reasons the shim emits documented; a unit test fails if the shim adds one the page lacks |
| 27 | Reason "threshold = N" for `tokens` | `Number(CFG.token_budget)` | | tokens | WRONG | `token_budget` is now the string `auto`: renders NaN | uses `capacity_model.token_budget.effective` |
| 28 | Reason "threshold = N" for others | `/gateway/config` | | tokens / s | OK | | each shows the setting and its tag |
| 29 | Outcome legend | static | | | OK | | Routing mix legend |
| 30 | Telemetry tile: tokens/s now | as row 16 | | | WRONG | duplicate of 16 | |
| 31 | Telemetry tile: TTFT | as row 17 | | | STALE | | |
| 32 | Telemetry tile: KV cache % | `latest.engine.kv_cache_pct` | | % | OK | | Summary + Capacity |
| 33 | Telemetry tile: engine running / waiting | `latest.engine.running/waiting` | | requests | OK | | Summary |
| 34 | Chart: tokens/s (prompt and generation, one axis) | `series.fast[].engine.prompt_tok_s, gen_tok_s` | polyline | tok/s | WRONG | one axis for two quantities that differ about 126x (hour mean 3,052 vs 24): generation is a flat line at zero | two charts |
| 35 | Chart: TTFT p50/p95 | `series.fast[].engine.ttft_*` | polyline | s | STALE | mostly gaps (row 17), no y-axis labels | chart with axes, gaps shown as gaps |
| 36 | Chart: inter-token latency | `engine.tpot_*` | | s | STALE | as row 35 | |
| 37 | Chart: "KV-cache % / lanes" | `engine.kv_cache_pct`, `gateway.inflight` | polyline | % and count | WRONG | percent and a count on one axis | KV cache and "work in flight" are separate charts |
| 38 | Charts: GPU 0 and GPU 1 (util %, temp °C, power % of cap) | `series.fast[].gpu[i]` | polyline | %, °C, % | WRONG | three units on one axis | three charts: busy, temperature, power in W, both cards each |
| 39 | Chart: remote-overflow share | `gateway.remote_share_pct` | polyline | % | OK | null in recent samples | Telemetry detail |
| 40 | Sparkline footers "last ~Nm" | `fast.length * 2 s` | | min | WRONG | 1,800 samples actually spanned 3,901 s (65 min), not 60: sample spacing is not 2.0 s | x-axis labelled from real timestamps |
| 41 | Chart y-scales | none | | | WRONG | no axis values at all, so no number could be read off a chart | axes, gridlines, hover values |
| 42 | Engine scrape indicator | `telemetry.latest.engine.ok/age_s/err` | | | OK | | Health + feed table |
| 43 | Per-client: requests, local, remote | `telemetry.per_client` | | requests | OK | window is since the process started; select said "since gateway start" | labelled "since this gateway process started" |
| 44 | Per-client: tokens out (exact, lower bound) | `tokens_out`, `tokens_out_lb` | `~` marker | tokens | OK | | |
| 45 | Per-client: avg wait, avg TTFT | `wait_avg_s`, `ttft_avg_s` | | s | OK | | |
| 46 | Per-client: errors | `errors` | | requests | OK | | |
| 47 | Per-client: est. cost | `cost_est_usd` | token counts x legacy prices | USD | OK | it is an estimate and said "rough"; ignores cache-hit pricing | tagged ESTIMATE |
| 48 | Per-client 24 h view: wait / TTFT / cost | `history/summary.per_client` | hard-coded `null` | | STALE | the log summary *has* `remote_cost_usd`, tokens in and cache tokens; the page showed `--` | filled |
| 49 | Host-role map | static `HOSTS` | | text | OK | hard-coded IPs | kept |
| 50 | Error feed | `telemetry.errors` | 60 rows | | OK | | Requests & history |
| 51 | History rows (time, client, route, preview, in→out, TTFT, duration, status) | `/gateway/history` | server paged | tokens, s | OK | | kept, with filters |
| 52 | History footer "logged N since restart" | `history.log.written` | | | OK | | kept |
| 53 | Recent requests "size (in→out)" | `events[].ptok → maxtok` | | tokens | WRONG | the "out" is the *requested ceiling*, not tokens produced | header says "Prompt → max out" |
| 54 | Recent requests caption "this process's uptime only" | `stats.events` | | | WRONG | the ring buffer is restored from disk on restart, so rows can predate the process | caption says so |
| 55 | Now list: "requests in flight" count | `lanes.active` | count where state = run | requests | OK | | |
| 56 | Now list: agent-lane state | `lanes.lanes[].status` | regex anchored at line start for RUNNING, DONE or BLOCKED followed by a bar | | WRONG | statuses that begin with a timestamp or `STALE |` fell through to RUNNING: finished and abandoned lanes were counted as "working" (9 of 9 shown running in the sample) | classified on the first 90 characters; lanes silent over 1 day hidden by default (1,184 lanes in the feed) |
| 57 | Now list: "overflows in ~Ns" | `config.local_wait_secs - active.waited` | | s | WRONG | ignores the work-class queue and deadlines (runner 900 s, background 600 s); the gateway reports the real `flow_expected_wait_s` per request | shows the gateway's expected wait |
| 58 | Now list: research / windows rows | `lanes.research`, `windows.windows` | | | OK | | kept |
| 59 | "longest still running" note | derived | | s | OK | | folded into list sorting |
| 60 | Estate watchdog checks | `lanes.health` | | | OK | | Health |
| 61 | Local model table (size GB, format, live) | `/gateway/models/local` | | GB | OK | | Settings |
| 62 | Settings form (33 fields) | `/gateway/config` | | various | OK | `token_budget` was a number input: cannot hold `auto` | text input with placeholder `auto`; all 33 fields kept |
| 63 | Telemetry payload per refresh | `/gateway/telemetry` | fetched whole every 2 s | bytes | WRONG | 2.29 MB per call (3,600 s of 2 s samples) = 1.15 MB/s per open tab, plus 0.41 MB `/gateway/lanes` every 2 s | trimmed server-side with `?fast=&slow=`; see section 4 |

**Counts: 63 items. 37 OK, 19 WRONG, 7 STALE.** The counts are tallied from the Verdict column of the table above.

## 2. What was fixed, and how

* Window and meaning on every number. Lifetime counters say "all time"; 15 min / 1 h / 24 h windows come from `/gateway/capacity.remote_use`.
* Configured vs live. Token budget, KV pool and prefill rate show their source. The page reads `capacity_model` (Lane GW) when the gateway publishes it, and says "configured, not read from the engine" when it does not.
* Effective routing mode (`/gateway/capacity.mode`, with the reasons) is the headline; the configured policy is separate.
* Latency shown three ways with their windows, because the engine's windowed quantile is empty most of the time.
* Charts: one unit per chart, real axes, hover values, 15-minute and 2-hour ranges.
* Process uptime (`stats.process_uptime`) distinguished from counter age (`stats.stats_since`): the only two server fields added, plus the `?fast=`/`?slow=` trim on `/gateway/telemetry`.
* Agent-lane status and the "overflows in" estimate no longer contradict the gateway.

## 3. Data the gateway already had that the old page never showed

| Data | Endpoint.field | Where it is now |
|---|---|---|
| Effective mode, why, mode history, time share per mode | `capacity.mode`, `why`, `mode_changes` | Routing |
| Planned offline window: holder, reason, remaining | `capacity.offline_window` | Summary, Routing |
| Spend today vs $25 cap, held, reserved, available, enforcement | `/gateway/spend`, `capacity.remote_use.spend_today` | Summary, Spend |
| Spend by client / reason / basis | `spend.attribution.groups` | Spend |
| Share of today's spend that was **not** metered from provider usage (basis `held-fallback`, `orphan-held`) | `spend.attribution.groups[].basis` | Spend (live: 58% of $21.95) |
| Decode tok/s total and per stream; prefill per request, engine aggregate, and the rate admission actually plans with | `capacity.throughput` | Throughput |
| Perf breaker state and reason | `stats.perf_breaker(_reason)` | banner, Health |
| Per-work-class queue, wait p50/p95, expected wait, refused, share, ceiling, deadline, demand | `capacity.queue`, `demand`, `config` | Work classes |
| Demand vs engine capacity, prefill backlog vs target | `capacity.pressure`, `stats.prefill_backlog_secs` | Capacity, Work classes |
| Prefix-affinity grants, hit rate grouped vs not | `capacity.affinity` | Work classes, Cache |
| Gateway cache-prediction accuracy and trust | `stats.cache_model` | Cache |
| Large cache mis-estimates | `telemetry.mis_estimates` | Cache (the settings hint told the operator to "check the mis-estimate feed", which was nowhere on the page) |
| Provider cache hit/miss tokens and cost per client | `history/summary.per_client` | Cache |
| Local-first outcomes: kept local, still remote and why | `stats.local_first` | Routing |
| `remote_while_local_had_headroom` (the defect signal), `gateway_chosen_remote` | `capacity.remote_use.windows` | Summary, Routing |
| Stall-brake-not-applied counts | `capacity.stall_brake_not_applied` | Routing |
| `cannot_measure`, capacity-model warnings | `capacity.cannot_measure`, `capacity_model.warnings` | banner |
| GPU clocks, fan, PCIe, VRAM, memory-controller busy | `stats.gpu`, `telemetry.series` | Health, Telemetry detail |
| Host CPU / RAM / disk | `telemetry.latest.host` | Health |
| Latency by class (admission wait, queue+prefill, decode) | `history/summary.latency_by_class` | Throughput |
| 24 h latency and outcome totals | `history/summary` | Throughput, Routing |
| Cumulative latency since engine start | `telemetry.percentiles` | Throughput |
| Ledger vs provider bill, failed calls charged, hours drifted | `spend.reconciliation` (Lane SL) | Spend |
| Dated price table: current period, age, stale flag | `spend.pricing` (Lane SL) | Spend |
| Process uptime | `stats.process_uptime` (new) | Health |
| Every configuration value | `/gateway/config` | Telemetry detail (read-only) |
| Model modalities (text-only) | `/v1/models` | Health |
| Per-feed freshness, size, latency | client | Telemetry detail |

## 4. Findings the page cannot fix (for the gateway owners)

1. (FIXED by Lane DB2, section 5) `capacity.remote_use.windows.*.by_class` is almost all `"?"` (407 of 410 remote requests in 15 min): the work class is only recorded on the local admission path, so the per-class split of *remote* use is unknowable from the gateway. The Work classes section therefore infers class from the client name and says so.
2. On 2026-10-02, 57-58% of the recorded spend was on a non-metered basis (`held-fallback`, an upper-bound hold: 319 calls averaging $0.039 vs $0.0009 for metered calls), and the ledger said $20.98 against a provider bill of $8.15. Lane SL's commit `2e89542375` (settle from actual usage at dated prices) fixes the cause. The page shows it either way: the "How the charges were computed" card flags any non-metered share over 10% of a day's spend of $1 or more, and the "Ledger vs the provider's own bill" card shows `spend.reconciliation` once the gateway publishes it.
3. (FIXED by Lane DB2, section 5) `/gateway/lanes` is 407 KB per call (1,184 agent-lane rows, 200 research rows), polled every 5 s by the new page, 81 KB/s per tab. It needs a server-side limit or filter.
4. (FIXED by Lane DB2, section 5) The windowed TTFT / inter-token quantiles are empty 80% of the time because they are per-2-s-window. A 60 s window would make "recent" numbers reliable.
5. Before this change the page transferred about 1.4 MB/s per open tab; with the trim it is about 0.19 MB/s (7x less), measured by serialising the same payload (150-sample recent tail 202 KB / 4 s, 450+139 long tail 697 KB / 30 s). Until the gateway is republished it ignores the trim and the page falls back to polling the full payload every 10 s.

## 5. Lane DB2 follow-ups (2026-10-03)

| Finding | Change in the gateway (`keepalive-shim.py`) | Change on the page |
|---|---|---|
| Work class only recorded on the local admission path (407 of 410 remote requests were class `?`) | `_route_completions` stamps `flow_class` on the live request right after the background/Halo test, before any routing branch, so alias, vision, forced, intent, size, local-offline, local-down, failover and rejected routes all reach `flow_note_route()` / `record_event()` with a class. Also: `stats.events[].cls`, `telemetry.per_client[].by_class`, `history/summary.per_class` and `.remote_by_reason` (from the on-disk log, a true 24 h), and `remote_use.windows.*.covered_s` | Work classes: "Who the paid requests came from" table by real class (15 min, 1 h, 24 h from the log); "Who is in each class" shows the recorded classes, not an inference; Recent requests has a Class column. Against an older gateway the table shows 100% untagged and says why |
| `/gateway/lanes` 407 KB per call | `?active=1&limit=N&max_age_s=S&research_limit=N`. `active=1` drops done lanes, lanes silent over a day and finished research. Every response has `summary` (totals by state and by age, research counts) so a trimmed reader knows what it did not get. No parameters = the full list, as before | The page polls `?active=1&limit=200` (61 KB, 7x smaller; 421 KB full); ticking "include finished" switches the feed to the full list. The lane-state test is `DashLib.laneStage` and the server mirror `lane_state()`; a test compares them |
| TTFT / inter-token quantiles empty 80% of the time | `lat_note_request()` keeps one tuple per finished streaming request; `/gateway/telemetry.windowed_latency` has 60 s, 5 min and 15 min windows, local and remote apart (n, TTFT p50/p95, gap p50/p95); every 2 s sample carries `gateway.ttft60_*`, `itl60_*`. Gap = decode time / (output tokens - 1), exact token counts only | TTFT tile and a new "Last 60 s" column of the Response time table use it (with n); the two latency charts plot the 60 s series |
| Overflow reasons | `remote_use.windows.*.by_reason` now lists every reason (was top 8). A test fails if a reason the page documents is no longer emitted (literal calls, the final overflow ladder, the stream watchdog) or the reverse | "Each reason's share, side by side": 15 min (routing ledger), 24 h (on-disk log), all time (persistent counters); a column that only reaches back to process start says so |

All 23 documented reasons are still emitted. On the live gateway the 24 h in-memory ring covered 6,439 of 17,568 requests of the log (it restarts with the process), which is why the 24 h column prefers the log.
