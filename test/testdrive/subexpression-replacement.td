# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

#
# Tests for the optimization described in https://github.com/MaterializeInc/materialize/pull/6196/
# Additional tests in test/sqllogictest/transform/predicate_reduction.slt
#

$ set-sql-timeout duration=125ms

# Remove both newlines, references to internal table identifiers, and "materialize.public" strings, all with a single regexp
$ set-regex match=(\s\(u\d+\)|\n|materialize\.public\.) replacement=

> CREATE TABLE t1 (col_null INTEGER, col_not_null INTEGER NOT NULL);
> CREATE DEFAULT INDEX on t1

> INSERT INTO t1 VALUES (1, 1);

# The simplest expression there could be

> EXPLAIN SELECT * FROM t1 WHERE col_null IS NULL AND (col_null IS NULL AND col_not_null = 5);
"Explained Query (fast path):  Filter (#0) IS NULL AND (#1 = 5)    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)Notices:  - Notice: Index t1_primary_idx on t1(#0{col_null}, #1{col_not_null}) is too wide to use for literal equalities `#1{col_not_null} = 5`.    Hint: If your literal equalities filter out many rows, create an index whose key exactly matches your literal equalities: (#1{col_not_null})."

> EXPLAIN SELECT * FROM t1 WHERE col_not_null = 1 AND (col_not_null = 1 AND col_null = 5);
"Explained Query (fast path):  Project (#0, #1)    ReadIndex on=t1 t1_primary_idx=[lookup value=(5, 1)]Used Indexes:  - t1_primary_idx (lookup)"

# NULL-able expressions are dedupped
> EXPLAIN SELECT * FROM t1 WHERE col_null = 1 AND (col_null = 1 AND col_not_null = 5);
"Explained Query (fast path):  Project (#0, #1)    ReadIndex on=t1 t1_primary_idx=[lookup value=(1, 5)]Used Indexes:  - t1_primary_idx (lookup)"

# OR/disjunction at the top level

> EXPLAIN SELECT * FROM t1 WHERE col_null IS NULL OR (col_null IS NULL AND col_not_null = 5);
"Explained Query (fast path):  Filter (#0) IS NULL    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

> EXPLAIN SELECT * FROM t1 WHERE col_null IS NULL OR col_null IS NULL OR (col_null IS NULL AND col_not_null = 5);
"Explained Query (fast path):  Filter (#0) IS NULL    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"


> EXPLAIN SELECT * FROM t1 WHERE col_null IS NULL OR (col_null IS NULL AND col_not_null = 5) OR (col_null IS NULL AND col_not_null = 6);
"Explained Query (fast path):  Filter (#0) IS NULL    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# OR/disjunction at the lower level

> EXPLAIN SELECT * FROM t1 WHERE col_null IS NULL AND (col_null IS NULL OR col_not_null = 5);
"Explained Query (fast path):  Filter (#0) IS NULL    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# Nested OR/disjunction

> EXPLAIN SELECT * FROM t1 WHERE col_null IS NULL OR (col_null IS NULL OR col_not_null = 5);
"Explained Query (fast path):  Filter ((#0) IS NULL OR (#1 = 5))    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# A more complex expression

> EXPLAIN SELECT * FROM t1 WHERE (col_not_null + 1 / col_not_null) = 5 AND ((col_not_null + 1 / col_not_null) = 5 AND col_null = 6);
"Explained Query (fast path):  Filter (#0 = 6) AND (5 = (#1 + (1 / #1)))    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)Notices:  - Notice: Index t1_primary_idx on t1(#0{col_null}, #1{col_not_null}) is too wide to use for literal equalities `#0{col_null} = 6`.    Hint: If your literal equalities filter out many rows, create an index whose key exactly matches your literal equalities: (#0{col_null}, (#1{col_not_null} + (1 / #1{col_not_null})))."

# More nesting

> EXPLAIN SELECT * FROM t1 WHERE col_not_null + col_not_null + col_not_null = 5 AND (col_not_null + col_not_null + col_not_null = 5);
"Explained Query (fast path):  Filter (5 = ((#1 + #1) + #1))    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# The common expression contains an AND/conjunction itself

> EXPLAIN SELECT * FROM t1 WHERE ((col_not_null > 3) AND (col_not_null < 5)) AND ((col_not_null > 3) AND (col_not_null < 5) OR col_not_null = 10);
"Explained Query (fast path):  Filter (#1 < 5) AND (#1 > 3)    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# The common expression contains an OR/disjunction

> EXPLAIN SELECT * FROM t1 WHERE ((col_not_null > 3) OR (col_not_null < 5)) OR ((col_not_null > 3) OR (col_not_null < 5));
"Explained Query (fast path):  Filter ((#1 < 5) OR (#1 > 3))    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# Use of a deterministic function

> EXPLAIN SELECT * FROM t1 WHERE col_not_null % 2 = 5 AND (col_not_null % 2 = 5 IS NULL);
"Explained Query (fast path):  Constant <empty>"

# This is not optimized
> EXPLAIN SELECT * FROM t1 WHERE (col_not_null % 2) = 1 AND (((col_not_null % 2) = 1) = TRUE);
"Explained Query (fast path):  Filter (1 = (#1 % 2))    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# Column used on both sides of the expression
> EXPLAIN SELECT * FROM t1 WHERE (col_not_null = col_not_null + 1) AND (col_not_null = col_not_null + 1);
"Explained Query (fast path):  Filter (#1 = (#1 + 1))    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# TODO (https://github.com/MaterializeInc/materialize/issues/6262):  Avoid simplifying mz_sleep.

> EXPLAIN SELECT * FROM t1
  WHERE mz_unsafe.mz_sleep(col_not_null) > mz_unsafe.mz_sleep(col_not_null)
  AND (mz_unsafe.mz_sleep(col_not_null) > mz_unsafe.mz_sleep(col_not_null) = true);
"Explained Query (fast path):  Project (#0, #1)    Filter (#2 > #2)      Map (mz_sleep(integer_to_double(#1)))        ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"

# IN list inside the expression

> EXPLAIN SELECT * FROM t1 WHERE col_not_null IN (2, 3) AND col_not_null IN (2, 3);
"Explained Query (fast path):  Filter ((#1 = 2) OR (#1 = 3))    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)Notices:  - Notice: Index t1_primary_idx on t1(#0{col_null}, #1{col_not_null}) is too wide to use for literal equalities `#1{col_not_null} IN (2, 3)`.    Hint: If your literal equalities filter out many rows, create an index whose key exactly matches your literal equalities: (#1{col_not_null})."

# Partial matches

> EXPLAIN SELECT * FROM t1 WHERE col_not_null IN (2, 3) AND col_not_null IN (3, 4);
"Explained Query (fast path):  Filter (#1 = 3)    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)Notices:  - Notice: Index t1_primary_idx on t1(#0{col_null}, #1{col_not_null}) is too wide to use for literal equalities `#1{col_not_null} = 3`.    Hint: If your literal equalities filter out many rows, create an index whose key exactly matches your literal equalities: (#1{col_not_null})."

> EXPLAIN SELECT * FROM t1 WHERE col_not_null IN (2, 3) AND col_not_null IN (4, 5);
"Explained Query (fast path):  Constant <empty>"

# Expression inside an IN list

# Optimized in AND/conjunctions

> EXPLAIN SELECT * FROM t1 WHERE col_not_null = 1 AND TRUE IN (col_not_null = 1, col_not_null = 2);
"Explained Query (fast path):  Filter (#1 = 1)    ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)Notices:  - Notice: Index t1_primary_idx on t1(#0{col_null}, #1{col_not_null}) is too wide to use for literal equalities `#1{col_not_null} = 1`.    Hint: If your literal equalities filter out many rows, create an index whose key exactly matches your literal equalities: (#1{col_not_null})."

# Not optimized in OR/disjunctions

> EXPLAIN SELECT * FROM t1 WHERE col_not_null = 1 OR TRUE IN (col_not_null = 1, col_not_null = 2);
"Explained Query (fast path):  Project (#0, #1)    Filter (#2 OR (#2 = true) OR (true = (#1 = 2)))      Map ((#1 = 1))        ReadIndex on=t1 t1_primary_idx=[*** full scan ***]Used Indexes:  - t1_primary_idx (*** full scan ***)"
