-- Copyright Materialize, Inc. and contributors. All rights reserved.
--
-- Use of this software is governed by the Business Source License
-- included in the LICENSE file at the root of this repository.
--
-- As of the Change Date specified in that file, in accordance with
-- the Business Source License, use of this software will be governed
-- by the Apache License, Version 2.0.

-- =============================================================================
-- Fixed income: a trading desk's live position report.
--
--   books        (static, 64)   8 business groups x 8 books
--   bonds        (static, 4096) wide reference data hashed from the bond id
--   rating_actions (1 per 30s)  the reference data that *does* change, rarely
--   trades       (1 per 100ms)  each with a book leg and an opposite Street leg
--   positions    net quantity per (book, bond), Street included
--   prices       every 100ms 1/50th of the bonds reprice, each every 5s
--   blotter      positions x books x bonds x prices: the wide row traders pivot
--
-- Unlike the other domains this one ticks at 100ms rather than per `moment`:
-- `tenths` expands the scaffold's indexed `seconds` into 100ms instants. Data
-- only lands sub-second if `default_timestamp_interval` is below 1s, or if
-- something else is writing to tables (every write advances all tables).
--
-- Invariant: for every bond, positions summed over all books *including
-- Street* are exactly zero, at every timestamp.
--
-- Prerequisites: scaffold.sql
-- Knobs: \set fi_retention '3 hours'   trades in window = 10/s x retention.
--                                      3h gives ~88k live positions.
--        \set fi_price_slots 50       each bond reprices every slots x 100ms.
--                                      10 reprices every bond every second,
--                                      about 5x the work downstream.
-- =============================================================================

\if :{?fi_retention}   \else \set fi_retention '3 hours' \endif
\if :{?fi_price_slots} \else \set fi_price_slots 50 \endif

CREATE SCHEMA IF NOT EXISTS materialize_demo;
SET search_path = materialize_demo;

-- -----------------------------------------------------------------------------
-- Static reference data
-- -----------------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS books AS
SELECT
    id::int                                                      AS id,
    'BOOK-' || lpad(id::text, 2, '0')                            AS book,
    (ARRAY['Rates','IG Credit','HY Credit','EM','Munis',
           'Securitized','Covered','Distressed'])[1 + id / 8]    AS business_group
FROM generate_series(0, 63) AS id;

-- Rating scale, by notch. `grade` is what traders pivot on.
CREATE VIEW IF NOT EXISTS rating_scale (notch, rating, grade) AS VALUES
    ( 0, 'AAA', 'AAA'), ( 1, 'AA+', 'AA'),  ( 2, 'AA',  'AA'),  ( 3, 'AA-', 'AA'),
    ( 4, 'A+',  'A'),   ( 5, 'A',   'A'),   ( 6, 'A-',  'A'),   ( 7, 'BBB+','BBB'),
    ( 8, 'BBB', 'BBB'), ( 9, 'BBB-','BBB'), (10, 'BB+', 'BB'),  (11, 'BB',  'BB'),
    (12, 'BB-', 'BB'),  (13, 'B+',  'B'),   (14, 'B',   'B'),   (15, 'B-',  'B');

-- 4,096 bonds. Every attribute is hashed from the id, so it never changes.
-- The one exception is the rating, which `rating_actions` moves below.
-- Byte budget on digest('bond:' || id):
--   [0]  currency (skewed: USD > EUR > GBP > ...)
--   [1]  sector mod 12
--   [2]  base rating notch mod 16
--   [3]  coupon (0.1% .. 8.0%)
--   [4]  base price, whole points (80 .. 119)
--   [5]  base price, fraction
--   [6]  years to maturity (1 .. 30)
--   [7]  seniority mod 3
--   [8]  day count mod 3
--   [9]  issue size (250M .. 2B)
--   [10..11] issuer id mod 1024
--   [12] callable (< 64)
CREATE VIEW IF NOT EXISTS bonds AS
SELECT
    id,
    'ZZ' || lpad((get_byte(h, 13) * 65536 + get_byte(h, 14) * 256 + get_byte(h, 15))::text, 9, '0') || mod(id, 10)
                                                                 AS isin,
    'BND-' || lpad(id::text, 4, '0')                             AS bond,
    'Issuer ' || lpad(mod(get_byte(h, 10) * 256 + get_byte(h, 11), 1024)::text, 4, '0')
                                                                 AS issuer,
    CASE WHEN get_byte(h, 0) < 110 THEN 'USD'
         WHEN get_byte(h, 0) < 170 THEN 'EUR'
         WHEN get_byte(h, 0) < 200 THEN 'GBP'
         WHEN get_byte(h, 0) < 220 THEN 'JPY'
         WHEN get_byte(h, 0) < 232 THEN 'CAD'
         WHEN get_byte(h, 0) < 242 THEN 'AUD'
         WHEN get_byte(h, 0) < 250 THEN 'CHF'
         ELSE 'SEK' END                                          AS currency,
    (ARRAY['Sovereign','Agency','Supranational','Financials','Industrials',
           'Utilities','Energy','Technology','Telecom','Consumer',
           'Healthcare','Real Estate'])[1 + mod(get_byte(h, 1), 12)] AS sector,
    mod(get_byte(h, 2), 16)                                      AS base_notch,
    (1 + mod(get_byte(h, 3), 80))::numeric / 10                  AS coupon,
    CASE WHEN get_byte(h, 3) < 128 THEN 2 ELSE 1 END             AS coupon_freq,
    (ARRAY['30/360','ACT/ACT','ACT/360'])[1 + mod(get_byte(h, 8), 3)] AS day_count,
    80 + mod(get_byte(h, 4), 40) + get_byte(h, 5)::numeric / 256 AS base_price,
    (DATE '2026-01-15' + (1 + mod(get_byte(h, 6), 30)) * INTERVAL '1 year')::date
                                                                 AS maturity,
    round((1 + mod(get_byte(h, 6), 30)) * 0.8, 1)                AS duration,
    CASE WHEN mod(get_byte(h, 6), 30) < 3  THEN '2Y'
         WHEN mod(get_byte(h, 6), 30) < 7  THEN '5Y'
         WHEN mod(get_byte(h, 6), 30) < 15 THEN '10Y'
         ELSE '30Y' END                                          AS benchmark,
    (ARRAY['Senior Secured','Senior Unsecured','Subordinated'])[1 + mod(get_byte(h, 7), 3)]
                                                                 AS seniority,
    (250 + mod(get_byte(h, 9), 8) * 250) * 1000000::bigint       AS issue_size,
    get_byte(h, 12) < 64                                         AS callable
FROM (SELECT id::int AS id, digest('bond:' || id::text, 'md5') AS h
      FROM generate_series(0, 4095) AS id);

CREATE DEFAULT INDEX IF NOT EXISTS ON books;
CREATE DEFAULT INDEX IF NOT EXISTS ON bonds;

-- -----------------------------------------------------------------------------
-- The 100ms clock
-- -----------------------------------------------------------------------------

-- Every 100ms instant in the retention window.
CREATE VIEW IF NOT EXISTS tenths AS
SELECT t FROM (
    SELECT generate_series(second, second + '900 milliseconds'::interval,
                           '100 milliseconds') AS t
    FROM seconds
)
WHERE mz_now() >= t AND mz_now() < t + :'fi_retention'::interval;

-- The most recent fi_price_slots 100ms instants. Each owns one price slot.
CREATE VIEW IF NOT EXISTS price_ticks AS
SELECT t, mod((EXTRACT(EPOCH FROM t) * 10)::bigint, :fi_price_slots) AS slot FROM (
    SELECT generate_series(second, second + '900 milliseconds'::interval,
                           '100 milliseconds') AS t
    FROM seconds
)
WHERE mz_now() >= t AND mz_now() < t + :fi_price_slots * '100 milliseconds'::interval;

-- -----------------------------------------------------------------------------
-- Slowly changing reference data: one rating action every 30 seconds.
-- Byte budget on digest('rating:' || second):
--   [0..1] bond_id mod 4096
--   [2]    direction (< 160 => downgrade, so credit drifts down)
-- A bond's current notch is its base plus all actions in the window, clamped.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS rating_actions AS
SELECT
    second                                                       AS acted_at,
    mod(get_byte(h, 0) + get_byte(h, 1) * 256, 4096)             AS bond_id,
    CASE WHEN get_byte(h, 2) < 160 THEN 1 ELSE -1 END            AS notches
FROM (SELECT second, digest('rating:' || second::text, 'md5') AS h FROM seconds)
WHERE mod(EXTRACT(EPOCH FROM second)::bigint, 30) = 0
  AND mz_now() >= second AND mz_now() < second + :'fi_retention'::interval;

CREATE VIEW IF NOT EXISTS bond_ratings AS
SELECT b.id AS bond_id, r.rating, r.grade
FROM bonds b
LEFT JOIN (SELECT bond_id, SUM(notches) AS moved FROM rating_actions GROUP BY bond_id) a
       ON a.bond_id = b.id
JOIN rating_scale r
  ON r.notch = GREATEST(0, LEAST(15, b.base_notch + COALESCE(a.moved, 0)));

CREATE DEFAULT INDEX IF NOT EXISTS ON bond_ratings;

-- -----------------------------------------------------------------------------
-- Prices: every 100ms the bonds with id % fi_price_slots = slot reprice at
-- their base +/- 1 point. Each bond keeps its price until its slot comes round.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS prices AS
SELECT
    b.id                                                         AS bond_id,
    p.t                                                          AS priced_at,
    round(b.base_price + (
        (get_byte(digest(b.id::text || p.t::text, 'md5'), 0) +
         get_byte(digest(b.id::text || p.t::text, 'md5'), 1) * 256
        )::numeric / 32768 - 1), 4)                              AS price
FROM bonds b JOIN price_ticks p ON mod(b.id, :fi_price_slots) = p.slot;

CREATE DEFAULT INDEX IF NOT EXISTS ON prices;

-- -----------------------------------------------------------------------------
-- Trades: one per 100ms.
-- Byte budget on digest(t):
--   [0..1] bond_id  mod 4096
--   [2]    book_id  mod 64
--   [3]    side     < 128 => buy
--   [4]    quantity (1 + mod 50) x 100k face
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS trades AS
SELECT
    t                                                            AS traded_at,
    mod(get_byte(h, 0) + get_byte(h, 1) * 256, 4096)             AS bond_id,
    mod(get_byte(h, 2), 64)                                      AS book_id,
    CASE WHEN get_byte(h, 3) < 128 THEN 1 ELSE -1 END            AS side,
    (1 + mod(get_byte(h, 4), 50)) * 100000                       AS quantity
FROM (SELECT t, digest(t::text, 'md5') AS h FROM tenths);

-- Positions, including the Street (book_id = -1) leg of every trade.
CREATE VIEW IF NOT EXISTS positions AS
SELECT book_id, bond_id, SUM(qty) AS quantity FROM (
    SELECT book_id, bond_id,  side * quantity AS qty FROM trades
    UNION ALL
    SELECT -1,      bond_id, -side * quantity AS qty FROM trades
)
GROUP BY book_id, bond_id;

CREATE DEFAULT INDEX IF NOT EXISTS ON positions;

-- -----------------------------------------------------------------------------
-- The blotter: the single wide table traders look at through pivots.
-- Street is excluded by the join to books. Flat positions are dropped.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS blotter AS
SELECT
    bk.business_group, bk.book,
    b.bond, b.isin, b.issuer, b.currency, b.sector, r.rating, r.grade,
    b.seniority, b.coupon, b.coupon_freq, b.day_count, b.maturity,
    b.benchmark, b.duration, b.issue_size, b.callable,
    p.quantity,
    pr.price,
    p.quantity * pr.price / 100                                  AS market_value,
    round(p.quantity * pr.price / 100 * b.duration / 10000, 2)   AS dv01
FROM positions p
JOIN books        bk ON bk.id = p.book_id
JOIN bonds        b  ON b.id  = p.bond_id
JOIN bond_ratings r  ON r.bond_id = p.bond_id
JOIN prices       pr ON pr.bond_id = p.bond_id
WHERE p.quantity <> 0;


-- -----------------------------------------------------------------------------
-- Validation:
--
-- Heartbeat (ticks every 100ms once default_timestamp_interval <= 100ms):
--   COPY (SUBSCRIBE (SELECT SUM(market_value) FROM blotter) WITH (progress)) TO STDOUT;
--
-- Invariant: every bond's positions net to zero across all books and Street.
-- Should always return 0.
--   SELECT COUNT(*) FROM (
--       SELECT bond_id FROM positions GROUP BY bond_id HAVING SUM(quantity) <> 0
--   );
--
-- Slowly changing reference data: recent rating actions.
--   SELECT * FROM rating_actions ORDER BY acted_at DESC LIMIT 5;
-- -----------------------------------------------------------------------------
