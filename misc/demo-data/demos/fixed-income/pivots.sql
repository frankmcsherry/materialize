-- Copyright Materialize, Inc. and contributors. All rights reserved.
--
-- Use of this software is governed by the Business Source License
-- included in the LICENSE file at the root of this repository.
--
-- As of the Change Date specified in that file, in accordance with
-- the Business Source License, use of this software will be governed
-- by the Apache License, Version 2.0.

-- =============================================================================
-- Pivots over the fixed_income blotter, built the way a data cube is:
-- aggregate once at the finest grain anyone pivots on, then derive every
-- tree level, cross-tab and trader screen from that small shared cube.
--
--   pivot_leaf     the cube: business_group x currency x sector x grade
--   mv_hist_leaf   a log-linear (HdrHistogram-style) histogram per cube cell,
--                  so median rolls up like a sum and retracts like one
--   pivot_tree     tree pivot: All > business group > currency > sector
--   median_tree    approximate median market value for every tree node
--   traders, expanded   per-trader UI state, as data
--   screens        every trader's visible tree rows, in ONE maintained view
--   pivot_group_by_ccy  2-D pivot: business groups down, currencies across
--
-- Prerequisites: scaffold.sql, domains/fixed_income.sql
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS materialize_demo;
SET search_path = materialize_demo;

-- -----------------------------------------------------------------------------
-- The wide blotter, arranged once. Without this index every view below would
-- rebuild the four-way join for itself.
-- -----------------------------------------------------------------------------
CREATE DEFAULT INDEX IF NOT EXISTS ON blotter;

-- -----------------------------------------------------------------------------
-- The cube. ~88k blotter rows collapse into at most 8 x 8 x 12 x 6 = 4,608
-- cells. Sum, count, min and max are carried. Average is sum / count.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS pivot_leaf AS
SELECT
    business_group, currency, sector, grade,
    COUNT(*)            AS n,
    SUM(quantity)       AS face,
    SUM(market_value)   AS market_value,
    SUM(dv01)           AS dv01,
    SUM(price)          AS sum_price,
    MIN(price)          AS min_price,
    MAX(price)          AS max_price
FROM blotter
GROUP BY business_group, currency, sector, grade;

CREATE DEFAULT INDEX IF NOT EXISTS ON pivot_leaf;

-- -----------------------------------------------------------------------------
-- Median, via histograms. Median is holistic: no fixed-size summary gives it
-- exactly. Fixed log-linear buckets (as in HdrHistogram) give it approximately,
-- and bucket counts are sums, so they merge up the tree and retract when a
-- position changes. Sketches like t-digest or KLL merge but cannot retract.
--
-- For |mv| in [2^e, 2^(e+1)) the bucket width is 2^(e-5): 32 buckets per
-- power of two, so a bucket midpoint is within 1/64 (~1.6%) of every value in
-- its bucket. Negative values (short positions) mirror positive ones, and zero
-- has its own bucket. A node holds at most ~32 buckets per power of two of
-- range, however many positions it covers.
--
-- NOTE: float8 math on purpose. Numeric log and pow are an order of magnitude
-- more expensive, and this runs for every blotter change.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS mv_hist_leaf AS
SELECT
    business_group, currency, sector, grade,
    CASE WHEN mv = 0 THEN 0
         ELSE CASE WHEN mv < 0 THEN -1 ELSE 1 END
              * (floor(abs(mv) / pow(2::float8, e - 5)) + 0.5)
              * pow(2::float8, e - 5)
    END                 AS mid,
    COUNT(*)            AS n
FROM (
    SELECT business_group, currency, sector, grade, market_value::float8 AS mv,
           CASE WHEN market_value = 0 THEN 0
                ELSE floor(ln(abs(market_value::float8)) / ln(2::float8)) END AS e
    FROM blotter
)
GROUP BY business_group, currency, sector, grade, mid;

CREATE DEFAULT INDEX IF NOT EXISTS ON mv_hist_leaf;

-- -----------------------------------------------------------------------------
-- Tree pivot: All > business_group > currency > sector. Every level is a
-- re-aggregation of the cube, not of the blotter. `path` identifies a node,
-- `parent` and `grandparent` let a trader's expanded set pick visible rows.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS pivot_tree AS
SELECT 0 AS level, 'All' AS path, NULL::text AS parent, NULL::text AS grandparent,
       'All' AS label,
       SUM(n) AS n, SUM(face) AS face, SUM(market_value) AS market_value,
       SUM(dv01) AS dv01, SUM(sum_price) / SUM(n) AS avg_price,
       MIN(min_price) AS min_price, MAX(max_price) AS max_price
FROM pivot_leaf
UNION ALL
SELECT 1, business_group, 'All', NULL, business_group,
       SUM(n), SUM(face), SUM(market_value), SUM(dv01), SUM(sum_price) / SUM(n),
       MIN(min_price), MAX(max_price)
FROM pivot_leaf GROUP BY business_group
UNION ALL
SELECT 2, business_group || '/' || currency, business_group, 'All', currency,
       SUM(n), SUM(face), SUM(market_value), SUM(dv01), SUM(sum_price) / SUM(n),
       MIN(min_price), MAX(max_price)
FROM pivot_leaf GROUP BY business_group, currency
UNION ALL
SELECT 3, business_group || '/' || currency || '/' || sector,
       business_group || '/' || currency, business_group, sector,
       SUM(n), SUM(face), SUM(market_value), SUM(dv01), SUM(sum_price) / SUM(n),
       MIN(min_price), MAX(max_price)
FROM pivot_leaf GROUP BY business_group, currency, sector;

-- The same levels, over histogram buckets.
CREATE VIEW IF NOT EXISTS hist_tree AS
SELECT 'All' AS path, mid, SUM(n) AS n FROM mv_hist_leaf GROUP BY mid
UNION ALL
SELECT business_group, mid, SUM(n) FROM mv_hist_leaf GROUP BY business_group, mid
UNION ALL
SELECT business_group || '/' || currency, mid, SUM(n)
FROM mv_hist_leaf GROUP BY business_group, currency, mid
UNION ALL
SELECT business_group || '/' || currency || '/' || sector, mid, SUM(n)
FROM mv_hist_leaf GROUP BY business_group, currency, sector, mid;

-- Median: the first bucket at which the running count reaches half the total.
-- Unindexed, for ad hoc queries. `screens` below computes it only for nodes
-- some trader can see, because a window function re-sorts its whole partition
-- on every change and nearly every node changes every 100ms.
CREATE VIEW IF NOT EXISTS median_tree AS
SELECT path, MIN(mid) AS median_mv
FROM (
    SELECT path, mid,
           SUM(n) OVER (PARTITION BY path ORDER BY mid) AS running,
           SUM(n) OVER (PARTITION BY path)              AS total
    FROM hist_tree
)
WHERE 2 * running >= total
GROUP BY path;

-- -----------------------------------------------------------------------------
-- Traders as data. Each trader's expand/collapse state is rows in a table,
-- and one maintained view serves every trader's visible rows. Adding a
-- trader adds rows, not a dataflow.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS traders (trader text);
CREATE TABLE IF NOT EXISTS expanded (trader text, path text);

CREATE VIEW IF NOT EXISTS screens AS
WITH visible AS (
    -- The root and business groups are always visible.
    SELECT t.trader, n.path
    FROM traders t, pivot_tree n
    WHERE n.level <= 1
    UNION ALL
    -- Currencies, under an expanded business group.
    SELECT e.trader, n.path
    FROM expanded e JOIN pivot_tree n ON n.parent = e.path
    WHERE n.level = 2
    UNION ALL
    -- Sectors, under an expanded currency whose business group is expanded too.
    SELECT e.trader, n.path
    FROM expanded e
    JOIN pivot_tree n  ON n.parent = e.path
    JOIN expanded   e2 ON e2.trader = e.trader AND e2.path = n.grandparent
    WHERE n.level = 3
),
visible_paths AS (SELECT DISTINCT path FROM visible),
visible_median AS (
    SELECT path, MIN(mid) AS median_mv
    FROM (
        SELECT h.path, h.mid,
               SUM(h.n) OVER (PARTITION BY h.path ORDER BY h.mid) AS running,
               SUM(h.n) OVER (PARTITION BY h.path)                AS total
        FROM hist_tree h JOIN visible_paths p ON p.path = h.path
    )
    WHERE 2 * running >= total
    GROUP BY path
)
SELECT v.trader, n.level, n.path, n.parent, n.label, n.n, n.face,
       n.market_value, n.dv01, n.avg_price, n.min_price, n.max_price,
       m.median_mv
FROM visible v
JOIN pivot_tree n ON n.path = v.path
LEFT JOIN visible_median m ON m.path = v.path;

CREATE INDEX IF NOT EXISTS screens_by_trader ON screens (trader);

-- -----------------------------------------------------------------------------
-- 2-D pivot: business groups down, currencies across. Also just the cube.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS pivot_group_by_ccy AS
SELECT
    business_group,
    SUM(market_value) FILTER (WHERE currency = 'USD') AS usd,
    SUM(market_value) FILTER (WHERE currency = 'EUR') AS eur,
    SUM(market_value) FILTER (WHERE currency = 'GBP') AS gbp,
    SUM(market_value) FILTER (WHERE currency = 'JPY') AS jpy,
    SUM(market_value) FILTER (WHERE currency = 'CAD') AS cad,
    SUM(market_value) FILTER (WHERE currency = 'AUD') AS aud,
    SUM(market_value) FILTER (WHERE currency = 'CHF') AS chf,
    SUM(market_value) FILTER (WHERE currency = 'SEK') AS sek,
    SUM(market_value)                                 AS total
FROM pivot_leaf
GROUP BY business_group;

CREATE DEFAULT INDEX IF NOT EXISTS ON pivot_group_by_ccy;

