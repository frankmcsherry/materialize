# Sweep: sweep-013 (2026-06-12 21:50)

- PLANDIFF test/sqllogictest/joins.slt (FAIL: output-failure=2 success=134 total=136) arrangements expected~19 actual~10
- PLANDIFF test/sqllogictest/subquery.slt (FAIL: output-failure=4 success=73 total=77) arrangements expected~28 actual~17
- PLANDIFF test/sqllogictest/aggregates.slt (FAIL: output-failure=2 success=116 total=118) arrangements expected~0 actual~0
- PLANDIFF test/sqllogictest/window_funcs.slt (FAIL: output-failure=5 success=667 total=672) arrangements expected~8 actual~4
- PLANDIFF test/sqllogictest/tpch_select.slt (FAIL: output-failure=14 success=37 total=51) arrangements expected~92 actual~47
- PLANDIFF test/sqllogictest/transform/predicate_reduction.slt (FAIL: output-failure=3 success=12 total=15) arrangements expected~0 actual~0
- PLANDIFF test/sqllogictest/transform/join_fusion.slt (FAIL: output-failure=4 success=21 total=25) arrangements expected~26 actual~12
- PASS test/sqllogictest/outer_join_lowering.slt
- PASS test/sqllogictest/cockroach/aggregate.slt
- PASS test/sqllogictest/cockroach/subquery_correlated.slt
- PASS test/sqllogictest/float.slt
- PASS test/sqllogictest/cockroach/distinct_on.slt

## Totals: pass=5 plandiff=7 resultdiff=0 panic=0
