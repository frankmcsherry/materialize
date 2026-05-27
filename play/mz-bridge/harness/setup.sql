-- Run against Materialize (psql "postgres://materialize@localhost:6875/materialize").
--
-- Creates a writable table and three materialized views that exercise the
-- different shapes the bridge must handle:
--   mv_accounts  - a keyed set (one row per id)
--   mv_names     - a true multiset (duplicate rows -> mz_diff > 1)
--   mv_balances  - an aggregate (sum per name)
-- All three keep an hour of history so the bridge can resume after a restart.

DROP MATERIALIZED VIEW IF EXISTS mv_accounts;
DROP MATERIALIZED VIEW IF EXISTS mv_names;
DROP MATERIALIZED VIEW IF EXISTS mv_balances;
DROP TABLE IF EXISTS accounts;

CREATE TABLE accounts (id int, name text, balance int);

CREATE MATERIALIZED VIEW mv_accounts
  WITH (RETAIN HISTORY FOR '1h') AS
  SELECT id, name, balance FROM accounts;

-- Projects away id, so common names produce duplicate rows (multiplicity).
CREATE MATERIALIZED VIEW mv_names
  WITH (RETAIN HISTORY FOR '1h') AS
  SELECT name FROM accounts;

CREATE MATERIALIZED VIEW mv_balances
  WITH (RETAIN HISTORY FOR '1h') AS
  SELECT name, sum(balance) AS total FROM accounts GROUP BY name;

INSERT INTO accounts VALUES
  (1, 'alice', 100),
  (2, 'bob',   200),
  (3, 'alice', 50);
