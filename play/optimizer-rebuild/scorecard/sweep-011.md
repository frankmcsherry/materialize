# Sweep: sweep-011 (2026-06-12 21:34)

- PLANDIFF test/sqllogictest/joins.slt (FAIL: output-failure=2 success=134 total=136) arrangements expected~19 actual~10
- PLANDIFF test/sqllogictest/subquery.slt (FAIL: output-failure=4 success=73 total=77) arrangements expected~28 actual~17
- PLANDIFF test/sqllogictest/aggregates.slt (FAIL: output-failure=3 success=115 total=118) arrangements expected~0 actual~0
- PLANDIFF test/sqllogictest/window_funcs.slt (FAIL: output-failure=11 success=661 total=672) arrangements expected~8 actual~4
- PLANDIFF test/sqllogictest/tpch_select.slt (FAIL: output-failure=16 success=35 total=51) arrangements expected~90 actual~46
- PLANDIFF test/sqllogictest/transform/predicate_reduction.slt (FAIL: output-failure=3 success=12 total=15) arrangements expected~0 actual~0
- PLANDIFF test/sqllogictest/transform/join_fusion.slt (FAIL: output-failure=4 success=21 total=25) arrangements expected~26 actual~12
- PLANDIFF test/sqllogictest/outer_join_lowering.slt (FAIL: output-failure=4 success=25 total=29) arrangements expected~26 actual~13
- PASS test/sqllogictest/cockroach/aggregate.slt
- PASS test/sqllogictest/cockroach/subquery_correlated.slt
- PASS test/sqllogictest/float.slt
- PASS test/sqllogictest/cockroach/distinct_on.slt

## Totals: pass=4 plandiff=8 resultdiff=0 panic=0
