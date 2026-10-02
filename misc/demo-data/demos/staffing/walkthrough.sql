-- Copyright Materialize, Inc. and contributors. All rights reserved.
--
-- Use of this software is governed by the Business Source License
-- included in the LICENSE file at the root of this repository.
--
-- As of the Change Date specified in that file, in accordance with
-- the Business Source License, use of this software will be governed
-- by the Apache License, Version 2.0.

-- Scripted walk through the staffing domain. Picks a front-line team with no
-- slack on some day, then requests and grants PTO, removes and backfills the
-- team's manager, and moves the team to another director, printing what the
-- director sees after each step. Every step writes to the manual_* tables,
-- so each run picks a fresh team. Run with psql -f; it needs psql for \gset.

\set ON_ERROR_STOP 1
SET search_path = materialize_demo;
\pset footer off

\echo
\echo '== Live? PTO requests by status (moves every few seconds), and the invariants (all 0).'
SELECT status, COUNT(*) FROM pto_requests GROUP BY status ORDER BY status;
SELECT * FROM staffing_invariants ORDER BY invariant;

-- A team that meets its requirement on some day with zero slack, even if every
-- pending request is granted, and a member who is in that day.
SELECT
    cov.position_id   AS team_seat,
    cov.manager_id,
    cov.manager_name,
    cov.day           AS pto_day,
    e.employee_id,
    e.name            AS employee_name,
    d.employee_id     AS director_id,
    d.name            AS director_name,
    d.position_id     AS director_seat
FROM staffing_coverage cov
JOIN employees e ON e.manager_id = cov.manager_id AND e.title = 'Specialist'
JOIN org_closure c ON c.position_id = cov.position_id
JOIN positions p ON p.position_id = c.ancestor_id AND p.level = 2
JOIN seat_holders d ON d.position_id = c.ancestor_id
WHERE cov.title = 'Manager'
  AND cov.available = cov.required
  AND cov.available_if_all_granted = cov.available
  AND cov.day > now() + INTERVAL '2 days'
  AND NOT EXISTS (
      SELECT 1 FROM pto_days x WHERE x.employee_id = e.employee_id AND x.day = cov.day)
ORDER BY cov.day, cov.position_id, e.employee_id
LIMIT 1 \gset

\echo
\echo '== Director' :director_name '(seat' :director_seat ') sees only their own org.'
SELECT COUNT(*) AS people_below FROM reports_to WHERE manager_id = :director_id;
SELECT status, COUNT(*) AS days FROM manager_alerts WHERE viewer_id = :director_id
GROUP BY status ORDER BY status;
SELECT day, levels_down, title, manager_name, status, required, available, available_if_all_granted
FROM manager_alerts WHERE viewer_id = :director_id
ORDER BY day, levels_down, position_id LIMIT 8;

\echo
\echo '== Team' :team_seat 'under' :manager_name 'on' :pto_day ': requirement met with no slack.'
SELECT day, seats, headcount, required, available, available_if_all_granted, status
FROM staffing_coverage WHERE position_id = :team_seat AND day = :'pto_day';

\echo
\echo '==' :employee_name 'asks for' :pto_day 'off. The day is now AT RISK, and the request shows in the director''s queue as breaking one requirement.'
INSERT INTO manual_pto_requests (employee_id, start_date, n_days)
VALUES (:employee_id, :'pto_day', 1);
SELECT request_id FROM pto_requests
WHERE employee_id = :employee_id AND decided_by = 'manual' AND status = 'requested'
ORDER BY request_id DESC LIMIT 1 \gset
SELECT request_id, name, start_date, levels_down, would_break
FROM manager_pto_queue WHERE viewer_id = :director_id AND request_id = :request_id;
SELECT day, manager_name, required, available FROM pto_impact WHERE request_id = :request_id;
SELECT day, manager_name, status, required, available, available_if_all_granted
FROM manager_alerts
WHERE viewer_id = :director_id AND position_id = :team_seat AND day = :'pto_day';

\echo
\echo '== Granted. The day is SHORT.'
INSERT INTO manual_pto_decisions (request_id, granted) VALUES (:request_id, true);
SELECT day, manager_name, status, required, available, available_if_all_granted
FROM manager_alerts
WHERE viewer_id = :director_id AND position_id = :team_seat AND day = :'pto_day';

\echo
\echo '==' :manager_name 'leaves. The team reports to the next filled seat up, and the empty seat costs a head every day.'
INSERT INTO manual_terminations (employee_id) VALUES (:manager_id);
SELECT e.name, e.manager_id, m.name AS acting_manager, m.title AS acting_title
FROM employees e JOIN employees m ON m.employee_id = e.manager_id
WHERE e.employee_id = :employee_id;
SELECT COUNT(*) AS alerts_for_departed_manager FROM manager_alerts WHERE viewer_id = :manager_id;
SELECT day, manager_name, status, required, available
FROM manager_alerts WHERE viewer_id = :director_id AND position_id = :team_seat
ORDER BY day;

\echo
\echo '== Backfilled. The new manager sees the team at once.'
INSERT INTO manual_hires (position_id, name) VALUES (:team_seat, 'Grace Okafor');
SELECT employee_id AS new_manager_id FROM employees
WHERE position_id = :team_seat AND source = 'manual' \gset
SELECT COUNT(*) AS direct_and_indirect_reports FROM reports_to WHERE manager_id = :new_manager_id;
SELECT day, manager_name, status, required, available
FROM manager_alerts WHERE viewer_id = :new_manager_id ORDER BY day;

\echo
\echo '== Reorg: the team moves under a senior manager in another director''s org.'
SELECT c.position_id AS new_parent
FROM org_closure c
JOIN positions p ON p.position_id = c.position_id AND p.level = 3
WHERE c.ancestor_id <> :director_seat AND c.depth = 1
ORDER BY c.position_id LIMIT 1 \gset
INSERT INTO manual_moves (position_id, new_parent_id) VALUES (:team_seat, :new_parent);
SELECT position_id, parent_id, reorged_at FROM org_tree WHERE position_id = :team_seat;
SELECT COUNT(*) AS people_below_old_director FROM reports_to WHERE manager_id = :director_id;
SELECT COUNT(*) AS old_director_sees_team
FROM manager_alerts WHERE viewer_id = :director_id AND position_id = :team_seat;
SELECT e.name AS new_director, COUNT(*) AS new_director_sees_team
FROM manager_alerts a
JOIN employees e ON e.employee_id = a.viewer_id
WHERE a.position_id = :team_seat AND e.title = 'Director'
GROUP BY e.name;

\echo
\echo '== Invariants, after all of that (all 0).'
SELECT * FROM staffing_invariants ORDER BY invariant;
