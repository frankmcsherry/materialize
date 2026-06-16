# Sweep: sweep-002 (2026-06-12 18:51)

- PASS test/sqllogictest/cockroach/aggregate.slt
- PASS test/sqllogictest/cockroach/subquery_correlated.slt
- PLANDIFF test/sqllogictest/joins.slt (FAIL: output-failure=7 success=129 total=136) arrangements expected~51 actual~29
- PLANDIFF test/sqllogictest/subquery.slt (FAIL: output-failure=5 success=72 total=77) arrangements expected~44 actual~29
- PLANDIFF test/sqllogictest/aggregates.slt (FAIL: output-failure=3 success=115 total=118) arrangements expected~2 actual~2
- PLANDIFF test/sqllogictest/window_funcs.slt (FAIL: output-failure=11 success=661 total=672) arrangements expected~10 actual~6
- PLANDIFF test/sqllogictest/tpch_select.slt (FAIL: output-failure=21 success=30 total=51) arrangements expected~90 actual~48
- PLANDIFF test/sqllogictest/outer_join_lowering.slt (FAIL: output-failure=4 success=25 total=29) arrangements expected~26 actual~13

## Totals: pass=2 plandiff=6 resultdiff=0 panic=0
