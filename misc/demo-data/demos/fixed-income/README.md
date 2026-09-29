# Fixed income: a live pivot report, worked end to end

A trading desk's position report, built on synthetic data that Materialize
generates for itself in SQL, following the "moments" technique from
[this blog post](https://github.com/frankmcsherry/blog/blob/master/posts/2024-05-19.md).
The scenario:

| The scenario | Where it lives here |
|---|---|
| Positions and prices change constantly, reference data rarely | `trades` and `prices` tick every 100ms. `rating_actions` re-rates one bond every 30s. |
| Up to ~100k rows, but wide | `blotter`: ~88k positions x 22 columns |
| Pivots by business group, currency, sector. Tree and 2-D. | `pivot_tree`, `pivot_group_by_ccy` |
| Sum, min, max, average, median | `pivot_leaf` (the cube), `mv_hist_leaf` (histograms for median) |
| Each trader sorts and expands differently, sees a screen's worth | `traders`, `expanded`, `screens`, `trader.py` |
| Updates about every 250ms | `freshness.py` |
| A screen traders click on | `board.py`, `board.html`, `board.sql` |
| Memory grows with every trader's customized view | `memory.py` |

The numbers below were measured on a laptop (Apple silicon, 10 cores, Docker
with 8GB) running `materialize/materialized` v26.43.0. Sections 9 onwards (the
board) were measured on an 18-core Apple silicon laptop, same image.

## Run it

The board, from nothing (Docker and Python 3 required, `psql` optional):

```sh
./run.sh             # then http://localhost:8765/?trader=alice, and ?trader=bob in another tab
```

Everything lands in the mz-demo-data skill's `materialize_demo` schema, so in
psql run `SET search_path = materialize_demo;` before the queries below.
`../../assets/teardown.sql` drops it all; the load is `IF NOT EXISTS`, so
changing a knob such as `FI_RETENTION` needs a teardown first.

To change the demo rather than just run it, see [PROMPT.md](PROMPT.md): the
scenario, the constants and where they live, what we learned, and references, set
out so a coding agent can rebuild it with your changes.

The earlier terminal version:

```sh
docker run -d --name mz -p 6875:6875 -p 6877:6877 -p 6878:6878 materialize/materialized:latest
./load.sh            # see the header for running it with docker exec instead of a local psql
pip install 'psycopg[binary]'
python3 trader.py alice
```

`trader.py` is a live screen redrawn four times a second. Arrow keys select,
enter expands or collapses, `s` cycles the sort column, `v` switches between
the tree and the currency cross-tab, `q` quits. Run it in two terminals as two
traders and expand different things.

`load.sh` sets `default_timestamp_interval` to 100ms, since without writes to
tables Materialize only advances time once a second. See [Freshness](#5-freshness).

## 1. The data

Everything is a view over `mz_now()`. The scaffold keeps a sliding window of
seconds, and this domain expands each second into ten 100ms instants
(`tenths`). Each instant is hashed into 16 bytes, and the bytes become a trade:
bond, book, side, size. Positions are the sum of trades still in the 3h window,
so a trade enters when its instant arrives and leaves when it ages out.

| Object | What it is | Size |
|---|---|---|
| `books` | 64 books in 8 business groups | static |
| `bonds` | 4,096 bonds with 16 reference columns (ISIN, issuer, currency, sector, coupon, maturity, ...) hashed from the id | static |
| `rating_actions` | one upgrade or downgrade every 30s | ~360 in window |
| `trades` | one every 100ms, each with a book leg and an opposite Street leg | ~108k in window |
| `prices` | every 100ms, 1/50th of the bonds reprice, so each bond every 5s | 4,096 |
| `blotter` | positions joined to books, bonds, ratings and prices | ~88k rows |

Every trade has an equal and opposite Street leg, so this holds at every
timestamp, whatever is in flight:

```sql
SELECT COUNT(*) FROM (
    SELECT bond_id FROM positions GROUP BY bond_id HAVING SUM(quantity) <> 0
);
-- 0
```

## 2. The cube

The report is a data cube in the sense of Gray et al., "Data Cube: A Relational
Aggregation Operator Generalizing Group-By, Cross-Tab, and Sub-Totals" (1996).
Aggregate once at the finest grain anyone pivots on, and derive every other
view from that.

`pivot_leaf` groups the ~88k blotter rows by business group x currency x sector
x grade, which is ~4,200 cells. The tree levels, the currency cross-tab, and
every trader's screen are re-aggregations of those cells, never of the rows.

* **Sum and count** re-aggregate exactly. **Average** is sum over count.
* **Min and max** re-aggregate from cells (a min of mins is the min), but
  maintaining them *under retractions* means keeping the values underneath, not
  just the current extreme. That is where the cube's memory goes. Of
  `pivot_leaf`'s 24MB, the sums account for 4MB and min/max for 19MB.
* **Median** does not re-aggregate at all. See the next section.

The 2-D pivot is also just the cube:

```sql
SELECT business_group, usd, eur, gbp, jpy, total FROM pivot_group_by_ccy;
```

## 3. Median, via histograms

Median is holistic: no fixed-size summary of a cell gives the median of a
merge. `mv_hist_leaf` keeps a histogram per cube cell instead, with
HdrHistogram-style log-linear buckets (32 per power of two, so a bucket's
midpoint is within 1.6% of every value in it). Bucket counts are sums. They
merge up the tree, and when a position changes they retract, which sketches
like t-digest and KLL cannot do.

The median of a tree node is the first bucket at which the running count
reaches half. Against the exact median, computed ad hoc from the blotter at
the same timestamp:

| node | exact | from histogram | error |
|---|---:|---:|---:|
| All | -85,290 | -84,992 | 0.35% |
| Covered | 110,763 | 111,616 | 0.77% |
| Distressed | 178,276 | 178,176 | -0.06% |
| Munis | -176,251 | -178,176 | -1.09% |
| Rates | -90,351 | -91,136 | -0.87% |

(market value, local currency units, abbreviated.)

Histogram size depends on how many rows a node covers:

| level | nodes | positions per node | buckets per node |
|---|---:|---:|---:|
| All | 1 | 88,397 | 465 |
| business group | 8 | 11,050 | 437 |
| currency | 64 | 1,381 | 280 |
| sector | 768 | 115 | 77 |
| leaf cell | 4,245 | 21 | 19 |

At the leaf a histogram is no smaller than the values. Near the top it is two
orders of magnitude smaller, and it is bounded by the value range, not the row
count, so it stays the same size as the book grows.

## 4. Traders as data

A trader's customization is rows: `traders` lists them and `expanded` lists the
tree nodes each has open. One indexed view, `screens`, computes every trader's
visible rows at once, and each trader subscribes to their slice:

```sql
INSERT INTO traders VALUES ('alice');
INSERT INTO expanded VALUES ('alice', 'Rates'), ('alice', 'Rates/USD');
COPY (SUBSCRIBE (SELECT * FROM screens WHERE trader = 'alice') WITH (PROGRESS)) TO STDOUT;
```

Expanding a node is an `INSERT`, collapsing is a `DELETE`, and the new rows
arrive through the subscription already open. Round trip from the write to
the rows on screen: 30 to 130ms, median 77ms.

`screens` also restricts the median computation to nodes some trader can see.
A window function re-sorts its partition on every change, and nearly every
node changes every 100ms, so computing medians for invisible nodes is
expensive work nobody reads.

Sorting and the choice between tree and cross-tab happen in the client, over
at most a few hundred rows.

## 5. Freshness

`default_timestamp_interval` (default 1s) is how often Materialize advances
time when nothing is written. It can be changed live:

```sql
-- as mz_system, on port 6877
ALTER SYSTEM SET default_timestamp_interval = '100ms';
```

Lag is wall-clock arrival minus the timestamp of the change, for every
timestamp that changed, measured by `freshness.py` with no table writes:

| interval | view | lag p50 | lag p99 | oracle writes/s | persist appends/s | consensus CAS/s | CAS latency |
|---|---|---:|---:|---:|---:|---:|---:|
| 1s | trader screen | 761ms | 1284ms | 1.2 | 25 | 33 | 3.3ms |
| 250ms | trader screen | 213ms | 353ms | 4.2 | 32 | 42 | 3.4ms |
| 100ms | trader screen | 76ms | 188ms | 10.2 | 43 | 60 | 1.9ms |
| 100ms | cube | 56ms | 136ms | 10.2 | 43 | 60 | 1.9ms |

Three things to know:

* **Writes advance time on their own.** With a client inserting at 10Hz,
  views were ~50ms fresh even at the 1s default. A system with prices and
  positions arriving constantly only needs this setting for quiet periods.
* **Sources have their own knob.** Kafka and CDC sources advance on their
  `TIMESTAMP INTERVAL`, which must lie within `min_timestamp_interval` and
  `max_timestamp_interval` (both 1s by default). Going below 1s means lowering
  `min_timestamp_interval` first. Not exercised here.
* **Each tick is round trips.** One oracle write and a persist append per
  table shard per tick. The CAS latencies above are against the Postgres
  inside the Docker image. In a cloud deployment consensus and the oracle are
  remote, and their latency is what to measure before promising 250ms.
  Materialized views also append to persist every tick. Indexes do not, which
  is why this demo uses only indexes.

## 6. Memory

The shared pipeline, whatever the number of traders:

| arrangement | MB |
|---|---:|
| `positions` (trades in window, netted) | 56 |
| `blotter` (the wide table, once) | 49 |
| `screens` (tree levels, medians, visible rows) | 28 to 42 |
| `pivot_leaf` (the cube) | 24 |
| `mv_hist_leaf` (histograms) | 11 |
| scaffold clock | 10 |

`memory.py` then adds N traders four ways and measures what they add. Each
trader gets a different tree (a different ordering of business group,
currency, sector and grade).

| approach | N | added MB | MB per trader | added cores |
|---|---:|---:|---:|---:|
| **grid**: own sorted copy of the wide blotter | 1 | 22 | 22 | 0.1 |
| | 8 | 179 | 22 | 1.3 |
| **direct**: own tree pivot from the blotter rows | 1 | 43 | 43 | 1.5 |
| | 4 | 186 | 46 | 4.5 |
| **cube**: own tree pivot from the shared cube | 1 | 7.4 | 7.4 | 0.4 |
| | 8 | 54 | 6.8 | 3.1 |
| **data**: rows in `traders` and `expanded` | 1 | 0.6 | 0.6 | 0.2 |
| | 8 | 15 | 1.9 | 0.5 |
| | 64 | 37 | 0.6 | 1.3 |
| | 256 | 61 | 0.24 | 2.0 |

* A trader's own copy of the rows, or their own pivot over the rows, costs in
  proportion to the rows: tens of MB each, which is the memory problem in the scenario.
* Pivoting from the shared cube cuts that by 6x, but each trader is still a
  dataflow doing work every tick, so CPU grows linearly.
* Traders as data grows with what they can see. 256 traders add 61MB in
  total, less than two traders computing their own pivot from the rows. The
  early per-trader cost is the medians of newly
  visible nodes, which is bounded by the tree (~840 nodes) and shared by
  everyone looking at them.

## 7. Knob sweep

`sweep.py` reloads the demo per configuration and records memory, CPU and
lag at each timestamp interval (`results/sweep.jsonl`). Four traders, 22
columns. Lag is p50/p99 ms on a trader's screen.

| retention, each bond reprices every | rows | row changes/s | arrangements | operator cores | lag at 1s | 250ms | 100ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1h, 5s | 33,626 | 6.7k | 91 MB | 1.8 | 715/1193 | 240/381 | 102/237 |
| 3h, 5s | 88,265 | 17.7k | 181 MB | 2.5 | 845/1276 | 253/420 | 77/215 |
| 3h, 1s | 88,278 | 88k | 225 MB | 6.3 | 1226/1716 | 377/621 | 587/776 |
| 3h, 0.5s | 88k | 177k | did not keep up | | | | |

At 1s repricing the laptop runs out of CPU, and the 100ms interval is worse
than 250ms because each tick adds work.

## 8. Wide rows

`wide.py` adds W reference columns per bond and holds them either once per
bond or joined into the ticking blotter (`results/wide.txt`).

| extra columns | once per bond | joined into blotter | bytes/row | cube lag p50/p99 |
|---:|---:|---:|---:|---:|
| 50 | 1.5 MB | +83 MB | 936 | 138/558 |
| 150 | 4.6 MB | +174 MB | 1,970 | 270/941 |
| 300 | 9.2 MB | +311 MB | 3,527 | 5,355/8,902 |
| 600 | 18.6 MB | +602 MB | 6,830 | no updates in 15s |

Every price tick retracts and re-inserts the whole row, so wide columns in the
ticking table cost on every tick. The runs went back to back on one instance.
Trust the pattern more than the thresholds.

## 9. The board

`board.py` puts a trader-style front end on the same data: a browser grid
with saved views as tabs, a pivot bar (rows, one dimension across, values), a
field chooser, a positions pane for the selected node, and a trade tape.

```sh
PIVOTS=board FI_CLUSTER_SIZE=200cc ./load.sh
python3 board.py            # http://localhost:8765/?trader=alice, and ?trader=bob in another tab
```

It serves a layout in one of two ways, and the strip above the grid says which
(PRE-MAINTAINED or LIVE QUERY), what the last click did and how long it took
to reach numbers. "SQL & timings" shows the query answering the screen and a
log of recent clicks. Tabs are tagged CUBE or LIVE from their current layout,
edits are kept per tab, and "+ New view" starts an empty one. In the field
chooser a trader can also type any SQL expression over the blotter's columns
as a grouping, and "run as live query" serves even a cube layout with a fresh
SUBSCRIBE, for comparison. Custom expressions go into the SQL as typed (one
expression, no `;` or comments): this is a local demo, not a safe interface.

* **Shared cube.** Business group, currency, sector, grade and tenor bucket,
  in any order and with any one of them across. `board.sql` generalizes
  `screens`: the layout is rows in `board_layout`, `board_across`,
  `board_expanded` and `board_measures`, and one indexed view, `board`, serves
  every trader. A click is a table write.
* **On demand.** Any other field (issuer, book, seniority, maturity year, ...).
  The server writes a GROUP BY over the blotter index for exactly the visible
  nodes and starts a new SUBSCRIBE on every click.

Sums are kept for every roll-up in use. Min, max and median keep the values
underneath, so `board` computes them only for nodes some trader who displays
them can see. Turning median on for one trader costs about a core.

Click to numbers, from the server receiving the click to the first update
showing its effect (`clicks.py`, `results/clicks.txt`):

| action | p50, two runs | p90, two runs |
|---|---:|---:|
| expand, shared cube | 196, 188 ms | 447, 360 ms |
| expand, shared cube, with min/max/median shown | 634, 604 ms | 721, 845 ms |
| expand, on demand (issuer > bond, new SUBSCRIBE) | 56, 59 ms | 86, 105 ms |
| switch to a new on-demand layout (new SUBSCRIBE) | 88, 104 ms | 120, 169 ms |

A cube click counts as done once the trader's subscription has passed a
timestamp read after the write, so it includes the write and the view's lag.

At this size a new SUBSCRIBE is quicker than a table write, because it reads
an index that is already up to date and does not wait on a write. What it costs is a dataflow per open
view: about 0.5 MB for an issuer pivot with sums only, 4 MB with min/max, and
about 32 MB and 1.5 cores for book > bond with min/max. The shared `board`
view was about 6 MB for all traders. These are single readings from the
board's own stats, not repeated trials, on a host where another Materialize
was busy.

### Replica size

The Docker image's `quickstart` cluster is 800cc, 16 timely workers. On this
data that is mostly coordination: every 100ms tick wakes all 16 workers for
every dataflow. With the board and two traders (`results/workers.txt`):

| size | workers | clusterd cores | operator cores | screen lag p50 | expand, cube | expand, cube + median | new live layout |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50cc | 1 | 0.9 | 0.8 | 136 ms | 158 ms | 5,873 ms | 29 ms |
| 100cc | 2 | 1.3 | 0.9 | 111 ms | 80 ms | 1,236 ms | 40 ms |
| **200cc** | **4** | **1.8** | **1.2** | **101 ms** | **102 ms** | **386 ms** | **40 ms** |
| 400cc | 8 | 3.0 | 1.4 | 97 ms | 90 ms | 347 ms | 106 ms |
| 800cc | 16 | 6.2 | 2.2 | 156 ms | 213 ms | 662 ms | 141 ms |

Four workers use under a third of the CPU of sixteen and are as fast or
faster everywhere. Fewer than four cannot keep up once median is on. The
click numbers in the table above were taken at 16 workers.

Where the extra CPU goes, from the replica's CPU profiler (`fgbreakdown.py`,
`results/profile.txt`, raw profiles in `results/profiles/`), in cores:

| bucket | 16 workers, 100ms | 16 workers, 1s | 4 workers, 100ms |
|---|---:|---:|---:|
| progress: pointstamp propagation (`Tracker`) | 1.06 | 0.22 | 0.22 |
| progress: broadcast between workers (`Progcaster`) | 0.56 | 0.16 | 0.04 |
| progress: other subgraph scheduling and frontiers | 0.42 | 0.12 | 0.07 |
| introspection logging | 0.87 | 0.29 | 0.14 |
| operators: arrangement merges | 0.93 | 0.41 | 0.39 |
| operators: everything else | 1.05 | 0.77 | 0.64 |
| worker loop, other threads | 0.93 | 0.33 | 0.17 |
| total | 5.81 | 2.31 | 1.66 |

From 4 to 16 workers, pointstamp propagation grows about in proportion to the
workers (every worker propagates the same changes), the broadcast grows 14x
(every worker sends to every other), introspection logging 6x and merges
2.4x. The 2,143 operators here see their frontiers move ten times a second.
Going to a 1s interval cut progress costs by 4x, not 10x: message counts
move pointstamps too.

`operators.py` (`results/operators.txt`) attributes the same thing from
introspection. At 16 workers, regions' own scheduling time (progress and
scheduling their children) is 3.1 cores against 2.6 in leaf operators; at 4
workers it is 0.4 against 1.1. Region schedulings run 820k/s against 330k/s
for leaf operators. A dataflow that receives no records, such as the
scaffold's `hours`, is still scheduled 22 times per worker per 100ms tick at
16 workers (10 at 4). Of 2,563 leaf operators, 19% are arrangements, reduces,
joins and top-ks, and they do 1.75 of the 2.55 cores. The rest are plumbing
(41%), introspection (29%: an `ArrangementSize` per arrangement, a
`LogOperatorHydration` per operator), and error-path operators that never see
a record (8%).

Turning introspection off (`ALTER CLUSTER quickstart SET (INTROSPECTION
INTERVAL = 0)`, `results/introspection.txt`) took the 16-worker replica from
6.25 to 3.51 cores and 4 workers from 2.36 to 1.73. Progress tracking at 16
workers fell from 1.99 to 1.38 cores, still 3.6x the 4-worker figure, so most
of the part that grows with workers is not introspection. One run per
configuration.

## Takeaways

1. Store the rows once, aggregate them once into a cube at the finest grain
   anyone pivots on, and derive every view from the cube.
2. Keep per-trader state as data, not as per-trader computation.
3. Min, max and median are where the memory goes, because retractions need
   the values underneath. Histograms make median a sum at bounded error.
4. Four updates a second is reachable. It is set by the timestamp interval,
   by source timestamp intervals, and by the round-trip latency of the
   deployment's consensus store and timestamp oracle.

## Caveats

* Market values are summed across currencies without FX conversion.
* Prices are base plus noise, not a random walk.
* The churn is set by `fi_price_slots` in the domain. At 10 (every bond
  reprices every second) the pipeline does about 5x the work.
* Process memory is much larger than arrangements. At 3h/5s the compute
  replica held 2.2GB RSS against 182MB of arrangements. Not yet explained.
* CPU is measured from Materialize's own operator timings, summed over leaf
  operators. Memory is arrangement bytes, not process RSS.

Raw outputs are in `results/`.
