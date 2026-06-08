-- Run against Materialize after the bridge is up, to watch changes flow through.
-- Each statement moves the views forward; the cohort fires one consistent moment.

-- new account
INSERT INTO accounts VALUES (4, 'carol', 75);

-- update a balance (delete + insert at one timestamp)
UPDATE accounts SET balance = 250 WHERE id = 2;

-- delete a row (mv_names multiplicity for 'alice' drops from 2 to 1)
DELETE FROM accounts WHERE id = 3;

-- bulk insert: several 'alice' rows to push mv_names multiplicity up
INSERT INTO accounts SELECT g, 'alice', g FROM generate_series(100, 105) AS g;
