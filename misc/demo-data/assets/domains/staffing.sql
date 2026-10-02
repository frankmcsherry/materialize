-- Copyright Materialize, Inc. and contributors. All rights reserved.
--
-- Use of this software is governed by the Business Source License
-- included in the LICENSE file at the root of this repository.
--
-- As of the Change Date specified in that file, in accordance with
-- the Business Source License, use of this software will be governed
-- by the Apache License, Version 2.0.

-- =============================================================================
-- Staffing: an org chart, people coming and going, and time off (PTO) against
-- per-manager staffing requirements, with each manager seeing only what rolls
-- up to them.
--
-- Demonstrates:
--   * recursion that stays live: who-reports-to-whom (WITH MUTUALLY RECURSIVE)
--     is maintained as people are hired and leave and as reorgs move seats
--   * tentative vs. confirmed state: requested PTO makes a day AT RISK,
--     granted PTO (and empty seats) make it SHORT
--   * managers as data: one index keyed by viewer serves every manager's
--     scoped view, so another manager is more rows, not another dataflow
--   * writable tables layered over the generated data, so a presenter can
--     request and decide PTO, hire, terminate, and move seats by hand
--   * per-login views: with RBAC on, a role granted only my_alerts sees only
--     its own org
--
-- Prerequisites: scaffold.sql, loaded with a one-week window so tenures and
-- PTO can span days:
--   \set retention '7 days'
--   \i scaffold.sql
-- Does not use common/people.sql: 256 identities are too few for 10k seats.
--
-- Shape:
--   positions        10,000 seats in a 7-way tree (CEO + 5 levels). Static,
--                    except that reorgs move a manager seat (and everything
--                    under it) to a new parent one level up.
--   employees        whoever holds each seat now. A hire takes a seat for
--                    2-7 days; a later hire into the same seat replaces them.
--                    manager_id is the holder of the nearest filled seat above.
--   reports_to       transitive closure of employees.manager_id.
--   pto_requests     0-2 per hire; requested -> granted (~80%) or denied.
--   staffing_coverage  per manager seat and day for the next 14 days:
--                    SHORT / AT RISK / OK against the seat's requirement.
--   pto_impact       pending requests that would make a requirement SHORT.
--   manager_alerts, manager_pto_queue   the above, keyed by viewer.
--   my_alerts, my_pto_queue, my_org     the same, for whoever is logged in.
--
-- Time: tick 1s. ~10k hires/day (one every ~9s), ~5k PTO requests/day,
-- ~42 reorgs/day. Every calendar day counts (a 24/7 operation).
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS materialize_demo;
SET search_path = materialize_demo;

-- -----------------------------------------------------------------------------
-- Static org chart.
--
-- Seat 0 is the CEO; the default parent of seat p is (p - 1) / 7. Levels start
-- at seats 0, 1, 8, 57, 400 and 2801, and seats 0..1428 have reports
-- ("manager seats"). Each manager seat must keep `required_fraction` of the
-- seats in its org (itself included) on duty every day: 80%..92%, salted per
-- seat so it is stable across loads.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS positions_static AS
SELECT
    p::int                                                       AS position_id,
    CASE WHEN p = 0 THEN NULL ELSE ((p - 1) / 7)::int END        AS default_parent_id,
    CASE WHEN p < 1    THEN 0
         WHEN p < 8    THEN 1
         WHEN p < 57   THEN 2
         WHEN p < 400  THEN 3
         WHEN p < 2801 THEN 4
         ELSE 5 END                                              AS level,
    p <= 1428                                                    AS is_manager_seat,
    0.80 + mod(get_byte(digest('staffing-req:' || p::text, 'md5'), 0)::int, 13)
         / 100.0                                                 AS required_fraction
FROM generate_series(0, 9999) AS p;

CREATE DEFAULT INDEX IF NOT EXISTS ON positions_static;

-- The seven VP seats name the departments; everything under a VP inherits it.
CREATE VIEW IF NOT EXISTS departments (position_id, department) AS VALUES
    (1, 'Engineering'),
    (2, 'Sales'),
    (3, 'Support'),
    (4, 'Operations'),
    (5, 'Finance'),
    (6, 'Marketing'),
    (7, 'People');

-- -----------------------------------------------------------------------------
-- Presenter tables. Rows here layer over the generated data:
--   manual_hires          takes the seat at hired_at, displacing the holder;
--                         employee_id = epoch milliseconds of hired_at
--   manual_terminations   the employee leaves at terminated_at; seat goes empty
--   manual_pto_requests   pending until a manual decision;
--                         request_id = epoch milliseconds of requested_at
--   manual_pto_decisions  overrides any generated decision (latest wins)
--   manual_moves          moves a seat at level 2..5 under a manager seat one
--                         level up; other rows are ignored, so no cycles
--   manager_logins        maps a database role to the employee it logs in as,
--                         for the my_* views at the bottom
-- Timestamps default to now(); future-dated rows take effect when they arrive.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS manual_hires (
    position_id int, name text, hired_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS manual_terminations (
    employee_id bigint, terminated_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS manual_pto_requests (
    employee_id bigint, start_date date, n_days int, requested_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS manual_pto_decisions (
    request_id bigint, granted bool, decided_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS manual_moves (
    position_id int, new_parent_id int, moved_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS manager_logins (
    role_name text, employee_id bigint);

-- -----------------------------------------------------------------------------
-- Reorgs: a manager seat at level 2..4 moves under a different seat one level
-- up, taking its whole subtree. Parents are always one level up, so the tree
-- cannot form a cycle. The latest move of a seat (generated or manual) wins;
-- when a generated one ages out of the window the seat returns to its default
-- parent.
--
-- Byte budget:
--   [0]    = 255   } reorg gate: 32 / 65,536 moments, ~42 a day
--   [1]    < 32    }
--   [2..3] seat that moves: 8 + (16-bit mod 1421), i.e. seats 8..1428
--   [4..5] new parent: uniform over the 7 / 49 / 343 seats one level up
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS reorgs_core AS
SELECT
    moment,
    position_id,
    CASE WHEN position_id < 57  THEN 1  + mod(pick, 7)
         WHEN position_id < 400 THEN 8  + mod(pick, 49)
         ELSE                        57 + mod(pick, 343) END     AS new_parent_id
FROM (
    SELECT
        moment,
        8 + mod(get_byte(random, 2) + get_byte(random, 3) * 256, 1421) AS position_id,
        get_byte(random, 4) + get_byte(random, 5) * 256          AS pick
    FROM random
    WHERE get_byte(random, 0) = 255 AND get_byte(random, 1) < 32
);

CREATE VIEW IF NOT EXISTS org_tree AS
SELECT
    p.position_id,
    COALESCE(r.new_parent_id, p.default_parent_id)               AS parent_id,
    r.moved_at                                                   AS reorged_at
FROM positions_static p
LEFT JOIN (
    SELECT DISTINCT ON (position_id) position_id, new_parent_id, moved_at
    FROM (
        SELECT position_id, new_parent_id, moment AS moved_at FROM reorgs_core
        UNION ALL
        SELECT m.position_id, m.new_parent_id, m.moved_at
        FROM manual_moves m
        JOIN positions_static s ON s.position_id = m.position_id
        JOIN positions_static np ON np.position_id = m.new_parent_id
        WHERE s.level >= 2 AND np.level = s.level - 1 AND np.is_manager_seat
          AND mz_now() >= m.moved_at
    )
    ORDER BY position_id, moved_at DESC
) r ON r.position_id = p.position_id;

CREATE INDEX IF NOT EXISTS org_tree_by_parent ON org_tree (parent_id);
CREATE INDEX IF NOT EXISTS org_tree_by_position ON org_tree (position_id);

-- Every (ancestor, seat) pair, including each seat with itself at depth 0.
CREATE VIEW IF NOT EXISTS org_closure AS
WITH MUTUALLY RECURSIVE
    closure (ancestor_id int, position_id int, depth int) AS (
        SELECT position_id, position_id, 0 FROM org_tree
        UNION ALL
        SELECT c.ancestor_id, t.position_id, c.depth + 1
        FROM closure c
        JOIN org_tree t ON t.parent_id = c.position_id
    )
SELECT ancestor_id, position_id, depth FROM closure;

CREATE INDEX IF NOT EXISTS org_closure_by_ancestor ON org_closure (ancestor_id);
CREATE INDEX IF NOT EXISTS org_closure_by_position ON org_closure (position_id);

CREATE VIEW IF NOT EXISTS positions AS
SELECT
    p.position_id,
    p.level,
    t.parent_id,
    CASE p.level
        WHEN 0 THEN 'CEO'
        WHEN 1 THEN 'VP'
        WHEN 2 THEN 'Director'
        WHEN 3 THEN 'Senior Manager'
        ELSE CASE WHEN p.is_manager_seat THEN 'Manager'
                  WHEN p.level = 4       THEN 'Senior Specialist'
                  ELSE                        'Specialist' END
    END                                                          AS title,
    COALESCE(d.department, 'Office of the CEO')                  AS department,
    p.is_manager_seat,
    p.required_fraction
FROM positions_static p
JOIN org_tree t ON t.position_id = p.position_id
LEFT JOIN (
    SELECT c.position_id, d.department
    FROM org_closure c
    JOIN departments d ON d.position_id = c.ancestor_id
) d ON d.position_id = p.position_id;

CREATE INDEX IF NOT EXISTS positions_by_position ON positions (position_id);

-- -----------------------------------------------------------------------------
-- Hires: one moment in ~8.5 hires someone into a seat.
--
-- Byte budget:
--   [0]    < 30   hire gate (~10k hires a day, one every ~9s)
--   [1..2] seat   16-bit mod 10,000
--   [3]    first name, mod 64
--   [4]    last name,  mod 64
--   [5]    tenure: 2 + mod 6 days (they may be replaced sooner)
--   [6]    PTO requests: < 154 none, < 230 one, else two (mean 0.5)
--   [7..]  free
-- employee_id is the hire moment's epoch seconds: unique, no 24-bit collisions.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS hires_core AS
SELECT
    moment,
    random,
    EXTRACT(EPOCH FROM moment)::bigint                           AS employee_id,
    mod(get_byte(random, 1) + get_byte(random, 2) * 256, 10000)  AS position_id,
    (ARRAY['Ada','Aiden','Amara','Andres','Anika','Ari','Bea','Beatriz',
           'Bilal','Camila','Chen','Chidi','Dana','Dmitri','Elena','Emeka',
           'Esther','Farah','Felix','Gabriel','Grace','Hana','Hugo','Ibrahim',
           'Ines','Isaac','Jamal','Jia','Jonas','Kai','Kavya','Kenji',
           'Lars','Leila','Lucia','Mateo','Maya','Mei','Mohammed','Nadia',
           'Naveen','Nia','Noah','Olga','Omar','Pablo','Priya','Quinn',
           'Rafael','Rosa','Sam','Sana','Sven','Tariq','Tess','Theo',
           'Uma','Valentina','Wei','Xavier','Yara','Yusuf','Zara','Zoe']
    )[1 + mod(get_byte(random, 3), 64)] || ' ' ||
    (ARRAY['Abara','Adeyemi','Alvarez','Andersen','Bauer','Bianchi','Brennan','Castillo',
           'Chandra','Chen','Costa','Dubois','Eriksen','Fischer','Fontaine','Garcia',
           'Gupta','Haddad','Hansen','Hoffman','Ibarra','Ito','Jansen','Kaur',
           'Kim','Kowalski','Kuznetsov','Larsen','Lee','Lindqvist','Lopez','Mahmoud',
           'Martin','Mensah','Moreau','Murphy','Nakamura','Nguyen','Novak','Okafor',
           'Olsen','Ortiz','Park','Patel','Petrov','Quinlan','Rahman','Reyes',
           'Rossi','Sato','Schmidt','Silva','Singh','Sorensen','Tanaka','Torres',
           'Usman','Varga','Walsh','Weber','Xu','Yamamoto','Zhang','Zielinski']
    )[1 + mod(get_byte(random, 4), 64)]                          AS name,
    moment + (2 + mod(get_byte(random, 5), 6)) * INTERVAL '1 day' AS tenure_ends_at,
    CASE WHEN get_byte(random, 6) < 154 THEN 0
         WHEN get_byte(random, 6) < 230 THEN 1
         ELSE 2 END                                              AS n_pto
FROM random
WHERE get_byte(random, 0) < 30;

-- Hires feed both seat holders and PTO. The index shares one hashing pass.
CREATE DEFAULT INDEX IF NOT EXISTS ON hires_core;

-- -----------------------------------------------------------------------------
-- Seat holders: the latest hire into each seat, while their tenure lasts and
-- they have not been terminated. Taking the latest hire *before* the tenure
-- filter is what makes a replaced employee stay gone.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS seat_holders AS
SELECT h.employee_id, h.position_id, h.name, h.hired_at, h.tenure_ends_at, h.source
FROM (
    SELECT DISTINCT ON (position_id)
        employee_id, position_id, name, hired_at, tenure_ends_at, source
    FROM (
        SELECT employee_id, position_id, name, moment AS hired_at, tenure_ends_at,
               'generated' AS source
        FROM hires_core
        UNION ALL
        SELECT (EXTRACT(EPOCH FROM hired_at) * 1000)::bigint, position_id, name,
               hired_at, hired_at + INTERVAL '7 days', 'manual'
        FROM manual_hires
    )
    WHERE mz_now() >= hired_at
    ORDER BY position_id, hired_at DESC, employee_id DESC
) h
WHERE mz_now() < h.tenure_ends_at
  AND h.employee_id NOT IN (
      SELECT employee_id FROM manual_terminations WHERE mz_now() >= terminated_at);

CREATE INDEX IF NOT EXISTS seat_holders_by_position ON seat_holders (position_id);
CREATE INDEX IF NOT EXISTS seat_holders_by_employee ON seat_holders (employee_id);

-- Employees, with manager_id = the holder of the nearest filled seat above.
-- When a manager's seat is empty, the next filled seat up is the acting manager.
CREATE VIEW IF NOT EXISTS employees AS
SELECT
    e.employee_id,
    e.name,
    e.position_id,
    p.title,
    p.department,
    m.manager_id,
    e.hired_at,
    e.tenure_ends_at                                             AS leaves_at,
    e.source
FROM seat_holders e
JOIN positions p ON p.position_id = e.position_id
LEFT JOIN (
    SELECT DISTINCT ON (c.position_id) c.position_id, h.employee_id AS manager_id
    FROM org_closure c
    JOIN seat_holders h ON h.position_id = c.ancestor_id
    WHERE c.depth > 0
    ORDER BY c.position_id, c.depth
) m ON m.position_id = e.position_id;

CREATE INDEX IF NOT EXISTS employees_by_employee ON employees (employee_id);
CREATE INDEX IF NOT EXISTS employees_by_manager ON employees (manager_id);

-- The transitive analogue of "manages": everyone under each manager, at any
-- depth. Managers are employees, so they appear on both sides.
CREATE VIEW IF NOT EXISTS reports_to AS
WITH MUTUALLY RECURSIVE
    r (manager_id bigint, employee_id bigint, depth int) AS (
        SELECT manager_id, employee_id, 1
        FROM employees
        WHERE manager_id IS NOT NULL
        UNION ALL
        SELECT r.manager_id, e.employee_id, r.depth + 1
        FROM r
        JOIN employees e ON e.manager_id = r.employee_id
    )
SELECT manager_id, employee_id, depth FROM r;

CREATE INDEX IF NOT EXISTS reports_to_by_manager ON reports_to (manager_id);

-- -----------------------------------------------------------------------------
-- PTO requests: 0-2 children per hire, by re-hashing the hire's bytes.
--
-- Byte budget (per request, md5(hire.random || n)):
--   [0]  filed b * 4 minutes after the hire (0..17 hours)
--   [1]  starts 1 + mod 14 days after the filing date
--   [2]  lasts 1 + mod 3 days
--   [3]  decided 1 + mod 12 hours after filing
--   [4]  < 205 granted (~80%), else denied
-- request_id = employee_id * 10 + n.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS pto_core AS
SELECT
    request_id,
    employee_id,
    requested_at,
    date_trunc('day', requested_at)
        + (1 + mod(get_byte(r, 1), 14)) * INTERVAL '1 day'       AS starts_on,
    1 + mod(get_byte(r, 2), 3)                                   AS n_days,
    requested_at + (1 + mod(get_byte(r, 3), 12)) * INTERVAL '1 hour' AS decided_at,
    get_byte(r, 4) < 205                                         AS granted
FROM (
    SELECT
        employee_id * 10 + n                                     AS request_id,
        employee_id,
        moment + get_byte(r, 0) * INTERVAL '4 minutes'           AS requested_at,
        r
    FROM (
        SELECT employee_id, moment, n, digest(random::text || n::text, 'md5') AS r
        FROM hires_core, generate_series(1, n_pto) AS n
    )
);

-- Lifecycle by temporal filter: a request appears at requested_at, is pending
-- until decided_at, and drops off once its last day is over. A manual
-- decision overrides the generated one. Only current employees' requests show.
CREATE VIEW IF NOT EXISTS pto_requests AS
WITH
    all_requests AS (
        SELECT request_id, employee_id, requested_at, starts_on, n_days,
               decided_at, granted, 'generated' AS source
        FROM pto_core
        UNION ALL
        SELECT (EXTRACT(EPOCH FROM requested_at) * 1000)::bigint, employee_id,
               requested_at, start_date::timestamptz, n_days,
               NULL::timestamptz, NULL::bool, 'manual'
        FROM manual_pto_requests
    ),
    phases AS (
        -- Pending: no decision yet.
        SELECT request_id, employee_id, requested_at, starts_on, n_days, source,
               NULL::bool AS granted
        FROM all_requests
        WHERE mz_now() >= requested_at
          AND mz_now() < COALESCE(decided_at, '2100-01-01'::timestamptz)
          AND mz_now() < starts_on + n_days * INTERVAL '1 day'
        UNION ALL
        -- Decided by the generator.
        SELECT request_id, employee_id, requested_at, starts_on, n_days, source,
               granted
        FROM all_requests
        WHERE mz_now() >= decided_at
          AND mz_now() < starts_on + n_days * INTERVAL '1 day'
    ),
    manual AS (
        SELECT DISTINCT ON (request_id) request_id, granted
        FROM manual_pto_decisions
        WHERE mz_now() >= decided_at
        ORDER BY request_id, decided_at DESC
    )
SELECT
    p.request_id,
    p.employee_id,
    e.name,
    e.position_id,
    p.starts_on::date                                            AS start_date,
    (p.starts_on + (p.n_days - 1) * INTERVAL '1 day')::date      AS end_date,
    p.n_days,
    p.requested_at,
    CASE COALESCE(m.granted, p.granted)
        WHEN true  THEN 'granted'
        WHEN false THEN 'denied'
        ELSE 'requested' END                                     AS status,
    CASE WHEN m.request_id IS NOT NULL THEN 'manual' ELSE p.source END AS decided_by
FROM phases p
JOIN seat_holders e ON e.employee_id = p.employee_id
LEFT JOIN manual m ON m.request_id = p.request_id;

CREATE INDEX IF NOT EXISTS pto_requests_by_request ON pto_requests (request_id);
CREATE INDEX IF NOT EXISTS pto_requests_by_employee ON pto_requests (employee_id);

-- -----------------------------------------------------------------------------
-- The staffing horizon: today and the next 13 days, rolling at midnight UTC.
-- `days` (from the scaffold) holds today as its latest row.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS horizon AS
SELECT (day + k * INTERVAL '1 day')::date AS day
FROM days, generate_series(0, 13) AS k
WHERE mz_now() < day + INTERVAL '1 day';

CREATE DEFAULT INDEX IF NOT EXISTS ON horizon;

-- One row per (request, day in the horizon) for requested or granted PTO.
CREATE VIEW IF NOT EXISTS pto_request_days AS
SELECT r.request_id, r.employee_id, r.position_id, r.status, h.day
FROM pto_requests r, generate_series(0, r.n_days - 1) AS k, horizon h
WHERE r.status IN ('requested', 'granted')
  AND h.day = (r.start_date + k * INTERVAL '1 day')::date;

-- Who is out on each day: granted wins over requested.
CREATE VIEW IF NOT EXISTS pto_days AS
SELECT employee_id, position_id, day, bool_or(status = 'granted') AS granted
FROM pto_request_days
GROUP BY employee_id, position_id, day;

CREATE INDEX IF NOT EXISTS pto_days_by_employee_day ON pto_days (employee_id, day);

-- -----------------------------------------------------------------------------
-- Coverage per manager seat and day.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS org_headcount AS
SELECT
    c.ancestor_id                                                AS position_id,
    COUNT(*)                                                     AS seats,
    COUNT(h.employee_id)                                         AS headcount
FROM org_closure c
LEFT JOIN seat_holders h ON h.position_id = c.position_id
GROUP BY c.ancestor_id;

CREATE INDEX IF NOT EXISTS org_headcount_by_position ON org_headcount (position_id);

CREATE VIEW IF NOT EXISTS org_out AS
SELECT
    c.ancestor_id                                                AS position_id,
    d.day,
    COUNT(*)                                                     AS out_any,
    COUNT(*) FILTER (WHERE d.granted)                            AS out_granted
FROM pto_days d
JOIN org_closure c ON c.position_id = d.position_id
GROUP BY c.ancestor_id, d.day;

CREATE INDEX IF NOT EXISTS org_out_by_position_day ON org_out (position_id, day);

CREATE VIEW IF NOT EXISTS staffing_requirements AS
SELECT
    p.position_id,
    p.title,
    p.department,
    h.employee_id                                                AS manager_id,
    h.name                                                       AS manager_name,
    o.seats,
    o.headcount,
    floor(p.required_fraction * o.seats)::int                    AS required
FROM positions p
JOIN org_headcount o ON o.position_id = p.position_id
LEFT JOIN seat_holders h ON h.position_id = p.position_id
WHERE p.is_manager_seat;

CREATE VIEW IF NOT EXISTS staffing_coverage AS
SELECT
    q.position_id,
    q.title,
    q.department,
    q.manager_id,
    q.manager_name,
    d.day,
    q.seats,
    q.headcount,
    q.required,
    COALESCE(x.out_granted, 0)                                   AS out_granted,
    COALESCE(x.out_any, 0) - COALESCE(x.out_granted, 0)          AS out_requested,
    q.headcount - COALESCE(x.out_granted, 0)                     AS available,
    q.headcount - COALESCE(x.out_any, 0)                         AS available_if_all_granted,
    CASE WHEN q.headcount - COALESCE(x.out_granted, 0) < q.required THEN 'SHORT'
         WHEN q.headcount - COALESCE(x.out_any, 0)     < q.required THEN 'AT RISK'
         ELSE 'OK' END                                           AS status
FROM staffing_requirements q
CROSS JOIN horizon d
LEFT JOIN org_out x ON x.position_id = q.position_id AND x.day = d.day;

CREATE INDEX IF NOT EXISTS staffing_coverage_by_position_day
    ON staffing_coverage (position_id, day);

-- Pending requests that would make a requirement SHORT if granted on their
-- own: the employee is not already out that day, and the requirement is met
-- with exactly zero slack.
CREATE VIEW IF NOT EXISTS pto_impact AS
SELECT
    rd.request_id,
    rd.employee_id,
    rd.day,
    cov.position_id,
    cov.manager_id,
    cov.manager_name,
    cov.required,
    cov.available
FROM pto_request_days rd
JOIN pto_days pd
  ON pd.employee_id = rd.employee_id AND pd.day = rd.day AND NOT pd.granted
JOIN org_closure c ON c.position_id = rd.position_id
JOIN staffing_coverage cov ON cov.position_id = c.ancestor_id AND cov.day = rd.day
WHERE rd.status = 'requested'
  AND cov.available = cov.required;

CREATE INDEX IF NOT EXISTS pto_impact_by_request ON pto_impact (request_id);

-- -----------------------------------------------------------------------------
-- Scoped by viewer: everything below the viewer's seat, transitively.
-- One index each; a manager's screen is a lookup or SUBSCRIBE on viewer_id.
-- -----------------------------------------------------------------------------

-- SHORT and AT RISK days for every manager seat in the viewer's org,
-- the viewer's own seat included.
CREATE VIEW IF NOT EXISTS manager_alerts AS
SELECT
    v.employee_id                                                AS viewer_id,
    c.depth                                                      AS levels_down,
    cov.position_id,
    cov.title,
    cov.manager_id,
    cov.manager_name,
    cov.day,
    cov.status,
    cov.required,
    cov.available,
    cov.available_if_all_granted
FROM seat_holders v
JOIN org_closure c ON c.ancestor_id = v.position_id
JOIN staffing_coverage cov ON cov.position_id = c.position_id
WHERE cov.status <> 'OK';

CREATE INDEX IF NOT EXISTS manager_alerts_by_viewer ON manager_alerts (viewer_id);

-- Pending requests from anyone below the viewer, with how many of the
-- viewer's org's requirement-days each would make SHORT. Breaks above the
-- viewer are not theirs to see, so they are not counted.
CREATE VIEW IF NOT EXISTS manager_pto_queue AS
SELECT
    v.employee_id                                                AS viewer_id,
    r.request_id,
    r.employee_id,
    r.name,
    c.depth                                                      AS levels_down,
    r.start_date,
    r.end_date,
    r.requested_at,
    COALESCE(b.would_break, 0)                                   AS would_break
FROM seat_holders v
JOIN org_closure c ON c.ancestor_id = v.position_id AND c.depth > 0
JOIN pto_requests r ON r.position_id = c.position_id AND r.status = 'requested'
LEFT JOIN (
    SELECT v2.employee_id AS viewer_id, i.request_id, COUNT(*) AS would_break
    FROM pto_impact i
    JOIN org_closure c2 ON c2.position_id = i.position_id
    JOIN seat_holders v2 ON v2.position_id = c2.ancestor_id
    GROUP BY v2.employee_id, i.request_id
) b ON b.viewer_id = v.employee_id AND b.request_id = r.request_id;

CREATE INDEX IF NOT EXISTS manager_pto_queue_by_viewer ON manager_pto_queue (viewer_id);

-- -----------------------------------------------------------------------------
-- Per-login views. Each filters by current_user through manager_logins, so a
-- role granted SELECT on these alone sees its own slice and nothing else.
-- That holds only with RBAC checks on (ALTER SYSTEM SET enable_rbac_checks
-- TO TRUE); the docker image ships with them off. current_user cannot be
-- materialized, so these stay plain views over the indexes above.
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS my_alerts AS
SELECT a.*
FROM manager_alerts a
JOIN manager_logins l ON l.employee_id = a.viewer_id
WHERE l.role_name = current_user;

CREATE VIEW IF NOT EXISTS my_pto_queue AS
SELECT q.*
FROM manager_pto_queue q
JOIN manager_logins l ON l.employee_id = q.viewer_id
WHERE l.role_name = current_user;

CREATE VIEW IF NOT EXISTS my_org AS
SELECT e.employee_id, e.name, e.title, e.department, r.depth AS levels_down,
       e.manager_id, e.hired_at
FROM reports_to r
JOIN employees e ON e.employee_id = r.employee_id
JOIN manager_logins l ON l.employee_id = r.manager_id
WHERE l.role_name = current_user;

-- -----------------------------------------------------------------------------
-- Invariants. Each row should read 0, at every timestamp, while hires,
-- departures, reorgs and PTO decisions land.
--
--   reports_to_agrees   reports_to (recursion over employees.manager_id) equals
--                       the seat closure joined to seat holders
--   headcount_rollup    a seat's org headcount = its own holder + the org
--                       headcounts of the seats directly under it
--   pto_rollup          the same, for who is out on each day
--   queue_scoped        every request in a viewer's queue is from someone who
--                       reports_to the viewer
--   alerts_scoped       every alert's manager is the viewer or reports_to them
-- -----------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS staffing_invariants AS
WITH
    via_seats AS (
        SELECT m.employee_id AS manager_id, e.employee_id
        FROM org_closure c
        JOIN seat_holders m ON m.position_id = c.ancestor_id
        JOIN seat_holders e ON e.position_id = c.position_id
        WHERE c.depth > 0
    ),
    via_manages AS (SELECT manager_id, employee_id FROM reports_to),
    headcount_rhs AS (
        SELECT position_id, SUM(n)::bigint AS headcount
        FROM (
            SELECT position_id, 1::bigint AS n FROM seat_holders
            UNION ALL
            SELECT t.parent_id, o.headcount
            FROM org_headcount o JOIN org_tree t ON t.position_id = o.position_id
            WHERE t.parent_id IS NOT NULL
        )
        GROUP BY position_id
    ),
    headcount_lhs AS (
        SELECT position_id, headcount FROM org_headcount WHERE headcount > 0
    ),
    out_rhs AS (
        SELECT position_id, day, SUM(n)::bigint AS out_any
        FROM (
            SELECT position_id, day, 1::bigint AS n FROM pto_days
            UNION ALL
            SELECT t.parent_id, x.day, x.out_any
            FROM org_out x JOIN org_tree t ON t.position_id = x.position_id
            WHERE t.parent_id IS NOT NULL
        )
        GROUP BY position_id, day
    ),
    out_lhs AS (SELECT position_id, day, out_any FROM org_out)
SELECT 'reports_to_agrees' AS invariant, COUNT(*) AS violations FROM (
    (SELECT * FROM via_seats EXCEPT ALL SELECT * FROM via_manages)
    UNION ALL
    (SELECT * FROM via_manages EXCEPT ALL SELECT * FROM via_seats)
)
UNION ALL
SELECT 'headcount_rollup', COUNT(*) FROM (
    (SELECT * FROM headcount_lhs EXCEPT ALL SELECT * FROM headcount_rhs)
    UNION ALL
    (SELECT * FROM headcount_rhs EXCEPT ALL SELECT * FROM headcount_lhs)
)
UNION ALL
SELECT 'pto_rollup', COUNT(*) FROM (
    (SELECT * FROM out_lhs EXCEPT ALL SELECT * FROM out_rhs)
    UNION ALL
    (SELECT * FROM out_rhs EXCEPT ALL SELECT * FROM out_lhs)
)
UNION ALL
SELECT 'queue_scoped', COUNT(*)
FROM manager_pto_queue q
WHERE NOT EXISTS (
    SELECT 1 FROM reports_to r
    WHERE r.manager_id = q.viewer_id AND r.employee_id = q.employee_id)
UNION ALL
SELECT 'alerts_scoped', COUNT(*)
FROM manager_alerts a
WHERE a.manager_id IS NOT NULL
  AND a.manager_id <> a.viewer_id
  AND NOT EXISTS (
    SELECT 1 FROM reports_to r
    WHERE r.manager_id = a.viewer_id AND r.employee_id = a.manager_id);

-- -----------------------------------------------------------------------------
-- Validation:
--
-- Heartbeat (PTO requests by status; moves every few seconds):
--   COPY (SUBSCRIBE (SELECT status, COUNT(*) FROM pto_requests GROUP BY status)
--         WITH (progress = true)) TO STDOUT;
--
-- Invariants (every row 0, always):
--   SELECT * FROM staffing_invariants;
--
-- One director's view: their org's red and amber days.
--   SELECT day, title, manager_name, status, required, available, available_if_all_granted
--   FROM manager_alerts
--   WHERE viewer_id = (SELECT employee_id FROM employees WHERE position_id = 8)
--   ORDER BY day, levels_down;
-- -----------------------------------------------------------------------------
