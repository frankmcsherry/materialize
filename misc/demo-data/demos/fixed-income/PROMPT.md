# Recipe: a live trade desk board

This file is the prompt that produces this demo. It builds on the
`mz-demo-data` skill (`.agents/skills/mz-demo-data`, which is `misc/demo-data/`):
point a coding agent at that skill and at this file, with one of:

* **Run it as it is.** No agent needed: `./run.sh`, then open
  http://localhost:8765/?trader=alice.
* **Change something and rebuild.** Edit the sections marked *Edit me*, then
  tell the agent: "Use the mz-demo-data skill. Read
  `misc/demo-data/demos/fixed-income/PROMPT.md`. The current code implements
  the previous version. Change it to match, keep the lessons in *What we
  learned*, and re-measure what the change affects."
* **Start over** (another asset class, another demo). Tell the agent to use the
  mz-demo-data skill to write a new domain, and to build from this file in a
  new directory under `demos/`. Keep *What we learned* and *References*, and
  rewrite the scenario.

The sections below are written to the agent.

---

## 1. The scenario (*Edit me*)

A representative workload: a fixed-income trading desk's live position report.

* A live, ticking report for traders. Positions and prices change constantly,
  reference data (currency, sector, rating, ...) rarely. These are combined
  into a single table, viewed through pivots.
* Freshness: traders expect updates about every 250ms.
* Pivots: group by business group, currency, bond sector. Tree pivots that
  expand and collapse, and 2-D pivots (currencies across). Sum, min, max,
  average, median. Each trader sorts and expands differently and sees a
  screen's worth.
* Scale: up to ~100k rows, but wide: hundreds of columns per row, on the
  order of 600.
* Memory is the usual pain: in systems that give each trader their own
  customized view, every view adds memory.

Treat this as a plausible workload, not a specification. Where it is vague
(what the wide columns are, how many rows change per second, how many traders
are on at once), make a reasonable choice, say what it is, and make it a knob.

## 2. Audience and framing (*Edit me*)

* **Who looks at it.** Engineers first, then people with trading backgrounds
  who judge whether it looks like a desk. Assume a skeptical audience.
* **What it has to show.**
  1. Many traders, each with their own tree, served from one maintained view,
     so memory barely moves as traders are added.
  2. Screens tick four times a second.
  3. A skeptic can name a grouping we could not have prepared for, click, and
     watch a live query start and time how long it takes to reach numbers.
     What is pre-built and what is started on the click must be obvious.
* **What it must not do.** Imitate a real vendor's product or branding. Use
  real data. Present synthetic numbers as measurements of anyone's system.
* **Deliverables.** The running board, the numbers behind every claim in
  `results/`, and the README walkthrough.

## 3. The data (*Edit me*)

Synthetic, generated inside Materialize from `mz_now()` with the "moments"
technique (reference 1), by `assets/domains/fixed_income.sql` on the
`mz-demo-data` scaffold (reference 2). Nothing is loaded from outside.

| Constant | Now | Where | What changing it does |
|---|---|---|---|
| Bonds | 4,096 | `fixed_income.sql`, `generate_series(0, 4095)` and every `mod 4096` | More distinct positions and reference rows. |
| Books / business groups | 64 / 8 | `books` view | More or fewer tree nodes at the top level. |
| Trade rate | one per 100ms | `tenths`, `trades` | Row churn in `positions`. |
| Retention (`FI_RETENTION`) | 3 hours | `load.sh` env | Live positions ≈ 10/s × retention: 1h ~34k, 3h ~88k rows. |
| Repricing (`FI_PRICE_SLOTS`) | 50: each bond every 5s | `load.sh` env | Row changes/s. 10 (every second) is ~5x the work and was CPU-bound on a laptop. 5 did not keep up. |
| Rating actions | one per 30s | `rating_actions` | The slowly changing reference data. |
| Reference columns | 16 per bond | `bonds` view | Blotter width. For widths on the order of 600 columns see `wide.py` and the wide-row finding below. |
| Prices | base ± 1 point, noise | `prices` | Not a random walk. |

Follow the skill's conventions: everything in the `materialize_demo` schema,
every `CREATE` `IF NOT EXISTS`, teardown is one `DROP SCHEMA`. The scaffold's
`retention` must cover the trade window, since the domain reads `seconds`.

Invariant that must keep holding: every bond's positions, Street included, net
to zero at every timestamp.

## 4. What gets built (*Edit me*)

| Knob | Now | Where | What changing it does |
|---|---|---|---|
| Cube dimensions ("hot") | business group, currency, sector, grade, tenor | `board.sql` `board_dims` and the five `CASE` lines in each view; `board.py` `HOT`/`BITS`; `board.html` `HOT` | Layouts using only these are served from the shared view with no per-trader dataflow. Each extra dimension doubles the possible roll-ups (masks) and grows the leaf. |
| Live-only fields | 13 reference, 3 ticking | `board.py` `REFERENCE`, `TICKING`; labels in `board.html` `DIMS` | What a skeptic can pick that is not pre-built. |
| Custom expressions | any SQL expression over blotter columns | `board.py` `valid_dim`; examples in `board.html` `EXAMPLES` | Local demo only: the SQL goes in as typed. |
| Measures | positions, face, MV, DV01, avg/min/max price, median MV | `board.sql`, `board.py` `MEASURES`, `board.html` `MEASURES` | Min, max and median are computed only for nodes some trader displaying them can see. |
| Median accuracy | 32 buckets per power of two (±1.6%) | `board.sql` `board_hist_leaf` (`e - 5`) | 128 buckets made the leaf histogram as big as the data. |
| Saved views (tabs) | 7 presets | `board.html` `PRESETS` | What the demo opens on. Tagged CUBE or LIVE from their layout. |
| Freshness | `default_timestamp_interval` 100ms | `load.sh` | 250ms gives ~250ms lag at less CPU. 1s is the Materialize default. |
| Replica size (`FI_CLUSTER_SIZE`) | 200cc, 4 workers | `run.sh` / `load.sh` | 800cc (16 workers, the image default) uses ~3x the CPU and is slower. Below 4 workers median can't keep up. |
| Redraw rate | 4 frames/s | `board.py` `/events?hz=` | The display cadence, independent of freshness. |
| Screen caps | 300 children per parent sent, 200 drawn, 20 across values | `board.py` `CHILD_MAX`, `ACROSS_MAX`; `board.html` `CHILD_SHOW` | Keep a silly pivot (bonds down, issuers across) from freezing the browser. |
| Wire format | the whole screen as JSON per frame | `board.py` `frame()` | See *Open directions*: diffs into a local store. |

## 5. What we learned (keep these)

Each came from a measurement in `results/` or a failure. A rebuild that
changes the scenario should still start from them.

* **Aggregate once.** Index the blotter once. Build a cube at the finest grain
  anyone pivots on, and derive every tree level, cross-tab and screen from it.
  Sum and count re-aggregate exactly; min and max need the values underneath
  under retractions, which is where cube memory goes; median does not
  re-aggregate at all.
* **Median as a sum.** Log-linear histograms (HdrHistogram-style) per cube cell
  make median a sum at bounded error. They retract, unlike t-digest or KLL. Use
  float8 for the bucket math: numeric `ln`/`pow` was 10x slower.
* **Traders as data.** A trader's layout is rows in tables (`board_layout`,
  `board_across`, `board_expanded`, `board_measures`). One indexed view serves
  every trader. A node is a mask plus a key with `'*'` in rolled-up dimensions,
  so "which nodes does this trader see" is equality joins.
* **Pay for holistic measures only where shown.** Median for visible nodes of
  traders who display it costs about a core. For everyone and every node it
  dominated CPU.
* **Indexes, not materialized views.** A materialized view writes to persist on
  every tick.
* **A new SUBSCRIBE on a click is fine at this size.** Against the blotter index
  a new GROUP BY reached numbers in ~30–100ms, faster than a cube click that
  writes a table (~70–200ms). What a live view costs is a dataflow while it's
  open: ~0.5 MB with sums only, tens of MB with min/max over fine groupings.
* **Time a cube click honestly.** Stop the clock only when the subscription's
  frontier passes a timestamp read after the write (`SELECT mz_now()` on the
  writer), or re-opening the same layout looks instant.
* **Rank by something that doesn't tick.** The positions pane ranks by face (only
  trades move it) and joins prices afterwards. Ranking by MV re-ranks on every
  price tick. The trade tape is a top-k over a 5-second temporal filter, not the
  whole trade window.
* **Workers.** On this data 16 workers were mostly coordination: ~6 cores
  against ~1.7 at 4 workers. See `results/workers.txt`, `results/profile.txt`
  and `results/operators.txt`. Introspection accounts for ~2.7 of those 6
  cores at 16 workers (`results/introspection.txt`); the board's stats read
  it, so keep it on for the demo.
* **Wide rows.** 600 extra reference columns cost 18.6 MB kept once per bond, but
  joined into the ticking blotter they stopped the pipeline: every price tick
  re-emits the whole row. If the width is bucketed risk (key-rate DV01 by
  tenor), store it as rows keyed by bucket and pivot tenor across in the UI.
* **UI lessons.** Say what is pre-built and what is live, on screen, on every
  click. Flash only moves over 0.25%, or everything flashes. Cap what a frame
  carries. Blank the grid on a new live layout instead of showing old rows
  under new labels. Resend a tab's layout when the event stream reconnects.
* **Environment traps.** `FETCH ... WITH (timeout)` on a SUBSCRIBE waits out the
  whole timeout: stream with `COPY (SUBSCRIBE ...) TO STDOUT`. Dropping a busy
  index can take minutes. Other Materialize instances may be listening on 6875
  or `localhost` may resolve to one: use `127.0.0.1` and a dedicated port. The
  Python relay tops out near 15k rows/s.

## 6. Open directions (*Edit me*: pick one to pursue)

* **Diffs into a local store.** Today `board.py` re-sends each screen as JSON
  four times a second. Send consolidated diffs per closed timestamp instead,
  into a browser store (TanStack DB), as the differential dataflow diagnostics
  console does (reference 5). Keep several SUBSCRIBEs consistent with each
  other with the cohort from `mz_bridge_recipe` (reference 6): grid, positions
  pane and tape at one timestamp, with `add` on navigation. With the cube's sums
  synced locally, switching tabs and expanding cube layouts become local
  re-aggregation. Measure the cube leaf's row changes/s first.
* **The wide columns.** Decide which are dimensions, which are measures and
  which only display, then pick the hot dimensions. If they are bucketed risk,
  model them as rows (see *Wide rows* above).

## 7. References

1. The "moments" technique for generating live data in SQL:
   https://github.com/frankmcsherry/blog/blob/master/posts/2024-05-19.md
2. The `mz-demo-data` skill this builds on: https://github.com/MaterializeInc/materialize/pull/38113
3. Gray et al., "Data Cube: A Relational Aggregation Operator Generalizing
   Group-By, Cross-Tab, and Sub-Totals" (1996): https://arxiv.org/abs/cs/0701155
4. HdrHistogram, for the log-linear buckets: http://hdrhistogram.org/
5. Differential dataflow's diagnostics console: it streams diffs over a
   WebSocket, one frame per closed timestamp, into TanStack DB collections in the
   browser, so tab switches draw from local state:
   https://github.com/TimelyDataflow/differential-dataflow/tree/master/diagnostics
   (`console/src/ws.ts` applies frames, `console/src/derive.ts` computes
   aggregates). TanStack DB: https://tanstack.com/db
6. `mz_bridge_recipe`: keeps several SUBSCRIBEs in a cohort and releases only
   consistent cuts, with `add`/`drop` for navigation:
   https://github.com/MaterializeInc/materialize/compare/main...frankmcsherry:materialize:mz_bridge_recipe?expand=1
   (`play/mz-bridge-recipe/`, see `DESIGN.md`).
7. FINOS Perspective, a widely used open-source streaming pivot grid, for what
   traders are used to: https://github.com/finos/perspective

## 8. Files

| File | What |
|---|---|
| `run.sh` | Everything from nothing: container, data, venv, board. |
| `load.sh` | Stands up the views on a running Materialize (`PIVOTS=board`, `FI_CLUSTER_SIZE`, ...). |
| `board.sql`, `board.py`, `board.html` | The board. |
| `pivots.sql`, `trader.py` | The earlier terminal version and its `screens` view. |
| `clicks.py` | Click-to-numbers latency, both paths. |
| `freshness.py`, `memory.py`, `sweep.py`, `wide.py` | The earlier measurements. |
| `fgbreakdown.py`, `operators.py` | CPU attribution: profiler buckets, and per-operator introspection. |
| `README.md` | The walkthrough with every number. |
| `results/` | Raw outputs behind the numbers. |
