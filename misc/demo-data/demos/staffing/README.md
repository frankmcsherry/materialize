# Staffing demo

Managers, their staffing requirements, and employees' time off, maintained
live as people come and go. Built on the `mz-demo-data` skill. The domain is
`assets/domains/staffing.sql`, with its byte budgets in the comments there.

| Goal | Where it shows |
|---|---|
| Managers see only what rolls up to them, transitively | `manager_alerts`, `manager_pto_queue` keyed by `viewer_id`; `my_*` views per login |
| Requirements that will not be met | `staffing_coverage`: `SHORT` with granted PTO and empty seats |
| Requested PTO is tentative and informs staffing | `AT RISK` = met now, not if all pending PTO is granted; `pto_impact` = the requirements one request would break |
| Employees come and go | ~10k hires a day into 10,000 seats; tenures end, seats empty, later hires replace holders |
| Managers are employees; manages is transitive | `employees.manager_id` (acting manager when the seat above is empty), `reports_to` by `WITH MUTUALLY RECURSIVE` |

## Load

The domain needs a one-week window, so the scaffold must be loaded fresh with
`retention` set. From `misc/demo-data`, against a local container:

```sh
mz() { docker exec -i mz-staffing psql -h localhost -p 6875 -U materialize "$@"; }
mz < assets/teardown.sql               # only if a scaffold is already loaded
mz -v retention='7 days' < assets/scaffold.sql
mz < assets/domains/staffing.sql
```

Hydration takes seconds. Steady state on a laptop container: ~140 MB of
arrangements, about one core busy, writes visible in a viewer's index within
~150 ms.

## Check it

```sql
SET search_path = materialize_demo;

-- Heartbeat: PTO requests by status change every few seconds.
COPY (SUBSCRIBE (SELECT status, COUNT(*) FROM pto_requests GROUP BY status)
      WITH (progress = true)) TO STDOUT;

-- Invariants: every row 0, at every timestamp.
SELECT * FROM staffing_invariants;
```

The invariants cross-check two independent routes to the same answer:
`reports_to` (recursion over `manager_id`) against the seat closure, org
totals against their direct children's totals, and every scoped row against
`reports_to`.

## Walk through it

```sh
mz < demos/staffing/walkthrough.sql
```

The script picks a front-line team that meets its requirement with no slack on
some day, then: one member requests that day off (AT RISK, and the director's
queue says it would break one requirement), the request is granted (SHORT),
the team's manager leaves (the team reports to the next seat up, every day
goes SHORT), a backfill hire sees the team at once, and the team moves to
another director's org. Each run picks a fresh team.

To watch a manager's screen change while you act, in a second terminal:

```sql
COPY (SUBSCRIBE (
    SELECT day, levels_down, manager_name, status, required, available,
           available_if_all_granted
    FROM manager_alerts WHERE viewer_id = <employee_id>)) TO STDOUT;
```

## Presenter tables

All timestamps default to `now()`; ids come from the queries above.

```sql
INSERT INTO manual_pto_requests (employee_id, start_date, n_days) VALUES (<id>, '2026-10-09', 2);
INSERT INTO manual_pto_decisions (request_id, granted) VALUES (<request_id>, true);
INSERT INTO manual_terminations (employee_id) VALUES (<id>);
INSERT INTO manual_hires (position_id, name) VALUES (<seat>, 'Grace Okafor');
INSERT INTO manual_moves (position_id, new_parent_id) VALUES (<seat>, <manager seat one level up>);
```

Generated decisions also land on their own, 1 to 12 hours after a request. A
manual decision overrides them. To undo every manual edit:

```sql
DELETE FROM manual_pto_requests; DELETE FROM manual_pto_decisions;
DELETE FROM manual_terminations; DELETE FROM manual_hires; DELETE FROM manual_moves;
```

## Per-login access

Materialize has no row-level security, but a view that filters on
`current_user` does the job once RBAC checks are on: a role granted `SELECT`
on `my_alerts` alone reads its own rows and is refused the views underneath.
The docker image ships with RBAC checks off, and cluster grants need the
system user:

```sh
mzsys() { docker exec -i mz-staffing psql -h localhost -p 6877 -U mz_system -d materialize "$@"; }
mzsys -c "ALTER SYSTEM SET enable_rbac_checks TO TRUE"
mzsys -c "CREATE ROLE grace" -c "GRANT USAGE ON CLUSTER quickstart TO grace"
```

```sql
-- as materialize
GRANT USAGE ON SCHEMA materialize_demo TO grace;
GRANT SELECT ON materialize_demo.my_alerts, materialize_demo.my_pto_queue,
                materialize_demo.my_org TO grace;
INSERT INTO materialize_demo.manager_logins VALUES ('grace', <employee_id>);
```

Then `docker exec -it mz-staffing psql -h localhost -p 6875 -U grace` sees its own rows in the `my_*` views and nothing else.
Simple probes (filters that divide by zero on other viewers' rows) did not
leak. That is an observation, not a guarantee.

## Tuning

* Requirements are 80% to 92% of the seats in a manager's org. At those
  settings about 3% of front-line team-days are SHORT and a typical director
  sees 5 to 50 alerts over two weeks. The fraction is `required_fraction` in
  `positions_static`.
* Org size, hire rate, tenure, PTO rate and decision delay are byte-budget
  choices, documented next to each `_core` view.
