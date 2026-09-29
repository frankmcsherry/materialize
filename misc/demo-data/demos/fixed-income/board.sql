-- Copyright Materialize, Inc. and contributors. All rights reserved.
--
-- Use of this software is governed by the Business Source License
-- included in the LICENSE file at the root of this repository.
--
-- As of the Change Date specified in that file, in accordance with
-- the Business Source License, use of this software will be governed
-- by the Apache License, Version 2.0.

-- =============================================================================
-- The live board: every trader's own pivot over the five "hot" dimensions,
-- in any order, with any one of them across, served from one maintained view.
--
--   board_leaf      the cube at the finest grain: group x ccy x sector x grade x tenor
--   board_cube      the roll-ups some trader's layout uses, sums only
--   board_layout    per trader: which dimension sits at each tree level
--   board_across    per trader: at most one dimension pivoted into columns
--   board_expanded  per trader: the tree nodes they have open
--   board_measures  per trader: which of min/max price and median MV they show
--   board           every trader's visible rows and cross-tab cells
--
-- A node is a cube row: a mask saying which dimensions it fixes, and a key
-- with '*' in the others. Its parent under a trader's layout is the same key
-- with the deepest fixed dimension set back to '*'. That turns "which nodes
-- does this trader see" into equality joins on keys.
--
-- Sums are maintained for every node in use. Min, max and median must keep
-- the values underneath, so they are computed only for nodes that some
-- trader who displays them can see. Turning median on is visible in CPU.
--
-- Pivots on any other column (issuer, book, seniority, ...) are not here. The
-- board server runs those as ad hoc SUBSCRIBEs against the blotter index.
--
-- Prerequisites: scaffold.sql, domains/fixed_income.sql. Replaces pivots.sql
-- for board.py, though both can be loaded together.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS materialize_demo;
SET search_path = materialize_demo;

CREATE DEFAULT INDEX IF NOT EXISTS ON blotter;

-- The five hot dimensions and their bits in a node's mask.
CREATE VIEW IF NOT EXISTS board_dims (dim, bit) AS VALUES
    ('business_group', 1), ('currency', 2), ('sector', 4), ('grade', 8), ('tenor', 16);

CREATE VIEW IF NOT EXISTS board_leaf AS
SELECT
    business_group, currency, sector, grade, benchmark AS tenor,
    COUNT(*)            AS n,
    SUM(quantity)       AS face,
    SUM(market_value)   AS market_value,
    SUM(dv01)           AS dv01,
    SUM(price)          AS sum_price,
    MIN(price)          AS min_price,
    MAX(price)          AS max_price
FROM blotter
GROUP BY business_group, currency, sector, grade, benchmark
-- ~9 positions per cell. Without the hint min/max is a 7-stage reduction.
OPTIONS (AGGREGATE INPUT GROUP SIZE = 64);

CREATE DEFAULT INDEX IF NOT EXISTS ON board_leaf;

-- Log-linear histogram of market value per leaf cell, as in pivots.sql:
-- 32 buckets per power of two, so a bucket midpoint is within ~1.6% of its
-- values. Bucket counts are sums, so they roll up and retract.
CREATE VIEW IF NOT EXISTS board_hist_leaf AS
SELECT
    business_group, currency, sector, grade, tenor,
    CASE WHEN mv = 0 THEN 0
         ELSE CASE WHEN mv < 0 THEN -1 ELSE 1 END
              * (floor(abs(mv) / pow(2::float8, e - 5)) + 0.5)
              * pow(2::float8, e - 5)
    END                 AS mid,
    COUNT(*)            AS n
FROM (
    SELECT business_group, currency, sector, grade, benchmark AS tenor,
           market_value::float8 AS mv,
           CASE WHEN market_value = 0 THEN 0
                ELSE floor(ln(abs(market_value::float8)) / ln(2::float8)) END AS e
    FROM blotter
)
GROUP BY 1, 2, 3, 4, 5, 6;

CREATE DEFAULT INDEX IF NOT EXISTS ON board_hist_leaf;

-- -----------------------------------------------------------------------------
-- Traders' layouts, as data. Everything a trader can change is a row here.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS board_layout (trader text, level int, dim text);
CREATE TABLE IF NOT EXISTS board_across (trader text, dim text);
CREATE TABLE IF NOT EXISTS board_expanded (trader text, mask int,
    business_group text, currency text, sector text, grade text, tenor text);
CREATE TABLE IF NOT EXISTS board_measures (trader text, measure text);

-- The mask of each tree level, per trader. Level 0 is the root.
CREATE VIEW IF NOT EXISTS board_levels AS
SELECT DISTINCT trader, 0 AS level, 0 AS mask FROM board_layout
UNION ALL
SELECT l.trader, l.level, SUM(d.bit)::int AS mask
FROM board_layout l
JOIN board_layout l2 ON l2.trader = l.trader AND l2.level <= l.level
JOIN board_dims d ON d.dim = l2.dim
GROUP BY l.trader, l.level;

-- The roll-ups some trader's layout uses: tree levels, and tree levels with
-- the across dimension added. At most 32, usually a handful.
CREATE VIEW IF NOT EXISTS board_masks AS
SELECT DISTINCT mask FROM (
    SELECT mask FROM board_levels
    UNION ALL
    SELECT l.mask | d.bit FROM board_levels l
    JOIN board_across a ON a.trader = l.trader
    JOIN board_dims d ON d.dim = a.dim
);

-- The roll-ups in use, from the leaf. Sums only: cheap at every mask.
CREATE VIEW IF NOT EXISTS board_cube AS
SELECT
    m.mask,
    CASE WHEN m.mask & 1  > 0 THEN business_group ELSE '*' END AS business_group,
    CASE WHEN m.mask & 2  > 0 THEN currency       ELSE '*' END AS currency,
    CASE WHEN m.mask & 4  > 0 THEN sector         ELSE '*' END AS sector,
    CASE WHEN m.mask & 8  > 0 THEN grade          ELSE '*' END AS grade,
    CASE WHEN m.mask & 16 > 0 THEN tenor          ELSE '*' END AS tenor,
    SUM(n) AS n, SUM(face) AS face, SUM(market_value) AS market_value,
    SUM(dv01) AS dv01, SUM(sum_price) / SUM(n) AS avg_price
FROM board_leaf, board_masks m
GROUP BY 1, 2, 3, 4, 5, 6;

CREATE INDEX IF NOT EXISTS board_cube_by_mask ON board_cube (mask);

-- Visible row nodes: the root, and every node whose parent is expanded.
CREATE VIEW IF NOT EXISTS board_visible AS
SELECT trader, 0 AS level, 0 AS mask,
       '*' AS business_group, '*' AS currency, '*' AS sector, '*' AS grade, '*' AS tenor
FROM board_levels WHERE level = 0
UNION ALL
SELECT lv.trader, lv.level, c.mask,
       c.business_group, c.currency, c.sector, c.grade, c.tenor
FROM board_cube c
JOIN board_levels lv ON lv.mask = c.mask AND lv.level > 0
JOIN board_levels pv ON pv.trader = lv.trader AND pv.level = lv.level - 1
JOIN board_expanded e
  ON e.trader = lv.trader AND e.mask = pv.mask
 AND e.business_group = CASE WHEN pv.mask & 1  > 0 THEN c.business_group ELSE '*' END
 AND e.currency       = CASE WHEN pv.mask & 2  > 0 THEN c.currency       ELSE '*' END
 AND e.sector         = CASE WHEN pv.mask & 4  > 0 THEN c.sector         ELSE '*' END
 AND e.grade          = CASE WHEN pv.mask & 8  > 0 THEN c.grade          ELSE '*' END
 AND e.tenor          = CASE WHEN pv.mask & 16 > 0 THEN c.tenor          ELSE '*' END;

-- Cross-tab cells: for a visible node with mask m and a trader's across
-- dimension with bit b, the cube rows at mask m | b that project back onto it.
CREATE VIEW IF NOT EXISTS board_cells AS
WITH wanted AS (
    SELECT DISTINCT v.trader, v.mask AS row_mask, v.mask | d.bit AS cell_mask, d.dim
    FROM board_visible v
    JOIN board_across a ON a.trader = v.trader
    JOIN board_dims d ON d.dim = a.dim
    WHERE v.mask & d.bit = 0
)
SELECT v.trader, v.level, v.mask,
       v.business_group, v.currency, v.sector, v.grade, v.tenor,
       CASE w.dim WHEN 'business_group' THEN c.business_group WHEN 'currency' THEN c.currency
                  WHEN 'sector' THEN c.sector WHEN 'grade' THEN c.grade ELSE c.tenor END AS across,
       c.n, c.face, c.market_value, c.dv01, c.avg_price
FROM board_cube c
JOIN wanted w ON w.cell_mask = c.mask
JOIN board_visible v
  ON v.trader = w.trader AND v.mask = w.row_mask
 AND v.business_group = CASE WHEN w.row_mask & 1  > 0 THEN c.business_group ELSE '*' END
 AND v.currency       = CASE WHEN w.row_mask & 2  > 0 THEN c.currency       ELSE '*' END
 AND v.sector         = CASE WHEN w.row_mask & 4  > 0 THEN c.sector         ELSE '*' END
 AND v.grade          = CASE WHEN w.row_mask & 8  > 0 THEN c.grade          ELSE '*' END
 AND v.tenor          = CASE WHEN w.row_mask & 16 > 0 THEN c.tenor          ELSE '*' END;

-- -----------------------------------------------------------------------------
-- Holistic measures, only where displayed. Each joins leaf rows to the visible
-- nodes they roll up into: project the leaf key onto every mask in use, and
-- keep the projections that match a node.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS board_extremes AS
WITH nodes AS (
    SELECT DISTINCT v.mask, v.business_group, v.currency, v.sector, v.grade, v.tenor
    FROM board_visible v JOIN board_measures m ON m.trader = v.trader
    WHERE m.measure IN ('min_price', 'max_price')
),
masks AS (SELECT DISTINCT mask FROM nodes)
SELECT n.mask, n.business_group, n.currency, n.sector, n.grade, n.tenor,
       MIN(l.min_price) AS min_price, MAX(l.max_price) AS max_price
FROM board_leaf l
CROSS JOIN masks m
JOIN nodes n
  ON n.mask = m.mask
 AND n.business_group = CASE WHEN m.mask & 1  > 0 THEN l.business_group ELSE '*' END
 AND n.currency       = CASE WHEN m.mask & 2  > 0 THEN l.currency       ELSE '*' END
 AND n.sector         = CASE WHEN m.mask & 4  > 0 THEN l.sector         ELSE '*' END
 AND n.grade          = CASE WHEN m.mask & 8  > 0 THEN l.grade          ELSE '*' END
 AND n.tenor          = CASE WHEN m.mask & 16 > 0 THEN l.tenor          ELSE '*' END
GROUP BY 1, 2, 3, 4, 5, 6
-- The root gathers every leaf cell, ~10k.
OPTIONS (AGGREGATE INPUT GROUP SIZE = 16384);

-- Median: the first bucket at which the running count reaches half.
CREATE VIEW IF NOT EXISTS board_median AS
WITH nodes AS (
    SELECT DISTINCT v.mask, v.business_group, v.currency, v.sector, v.grade, v.tenor
    FROM board_visible v JOIN board_measures m ON m.trader = v.trader
    WHERE m.measure = 'median_mv'
),
masks AS (SELECT DISTINCT mask FROM nodes),
hist AS (
    SELECT n.mask, n.business_group, n.currency, n.sector, n.grade, n.tenor,
           h.mid, SUM(h.n) AS n
    FROM board_hist_leaf h
    CROSS JOIN masks m
    JOIN nodes n
      ON n.mask = m.mask
     AND n.business_group = CASE WHEN m.mask & 1  > 0 THEN h.business_group ELSE '*' END
     AND n.currency       = CASE WHEN m.mask & 2  > 0 THEN h.currency       ELSE '*' END
     AND n.sector         = CASE WHEN m.mask & 4  > 0 THEN h.sector         ELSE '*' END
     AND n.grade          = CASE WHEN m.mask & 8  > 0 THEN h.grade          ELSE '*' END
     AND n.tenor          = CASE WHEN m.mask & 16 > 0 THEN h.tenor          ELSE '*' END
    GROUP BY 1, 2, 3, 4, 5, 6, 7
)
SELECT mask, business_group, currency, sector, grade, tenor, MIN(mid) AS median_mv
FROM (
    SELECT *,
           SUM(n) OVER (PARTITION BY mask, business_group, currency, sector, grade, tenor
                        ORDER BY mid) AS running,
           SUM(n) OVER (PARTITION BY mask, business_group, currency, sector, grade, tenor) AS total
    FROM hist
)
WHERE 2 * running >= total
GROUP BY 1, 2, 3, 4, 5, 6;

-- Everything a trader's screen needs: row nodes (across = '') and cross-tab
-- cells. Min, max and median are NULL where the trader has not asked for them.
CREATE VIEW IF NOT EXISTS board AS
SELECT v.trader, v.level, v.mask,
       v.business_group, v.currency, v.sector, v.grade, v.tenor,
       '' AS across,
       c.n, c.face, c.market_value, c.dv01, c.avg_price,
       x.min_price, x.max_price, m.median_mv
FROM board_visible v
JOIN board_cube c
  ON c.mask = v.mask AND c.business_group = v.business_group AND c.currency = v.currency
 AND c.sector = v.sector AND c.grade = v.grade AND c.tenor = v.tenor
LEFT JOIN board_extremes x
  ON x.mask = v.mask AND x.business_group = v.business_group AND x.currency = v.currency
 AND x.sector = v.sector AND x.grade = v.grade AND x.tenor = v.tenor
LEFT JOIN board_median m
  ON m.mask = v.mask AND m.business_group = v.business_group AND m.currency = v.currency
 AND m.sector = v.sector AND m.grade = v.grade AND m.tenor = v.tenor
UNION ALL
SELECT trader, level, mask, business_group, currency, sector, grade, tenor, across,
       n, face, market_value, dv01, avg_price, NULL::numeric, NULL::numeric, NULL::float8
FROM board_cells;

CREATE INDEX IF NOT EXISTS board_by_trader ON board (trader);

