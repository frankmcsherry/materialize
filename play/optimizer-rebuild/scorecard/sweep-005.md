# Sweep: sweep-005 (2026-06-12 20:01)

- PLANDIFF test/sqllogictest/joins.slt (FAIL: output-failure=6 success=130 total=136) arrangements expected~47 actual~27
- PLANDIFF test/sqllogictest/subquery.slt (FAIL: output-failure=5 success=72 total=77) arrangements expected~44 actual~29
- PLANDIFF test/sqllogictest/aggregates.slt (FAIL: output-failure=3 success=115 total=118) arrangements expected~2 actual~2
- PLANDIFF test/sqllogictest/window_funcs.slt (FAIL: output-failure=11 success=661 total=672) arrangements expected~10 actual~6
- PLANDIFF test/sqllogictest/tpch_select.slt (FAIL: output-failure=20 success=31 total=51) arrangements expected~90 actual~49
- PLANDIFF test/sqllogictest/transform/predicate_reduction.slt (FAIL: output-failure=3 success=12 total=15) arrangements expected~0 actual~0
- PLANDIFF test/sqllogictest/transform/join_fusion.slt (FAIL: output-failure=4 success=21 total=25) arrangements expected~28 actual~14
- PLANDIFF test/sqllogictest/outer_join_lowering.slt (FAIL: output-failure=4 success=25 total=29) arrangements expected~26 actual~13
- PASS test/sqllogictest/cockroach/aggregate.slt
- PASS test/sqllogictest/cockroach/subquery_correlated.slt
- PASS test/sqllogictest/float.slt
- PASS test/sqllogictest/cockroach/distinct_on.slt

## Totals: pass=4 plandiff=8 resultdiff=0 panic=0
