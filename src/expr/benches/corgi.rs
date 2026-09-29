// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Corgi functions evaluated a row at a time, as a `MirScalarExpr` evaluates
//! them, against the same function evaluated over the whole batch at once, and
//! against the equivalent built-in expression.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mz_expr::func::{AddInt64, CorgiFunc};
use mz_expr::{Eval, MirScalarExpr, VariadicFunc};
use mz_repr::{Datum, RowArena, SqlScalarType};

const ROWS: i64 = 10_000;

fn corgi_call(func: &CorgiFunc) -> MirScalarExpr {
    MirScalarExpr::CallVariadic {
        func: VariadicFunc::Corgi(func.clone()),
        exprs: (0..func.arg_types.len())
            .map(MirScalarExpr::column)
            .collect(),
    }
}

fn bench(
    c: &mut Criterion,
    name: &str,
    rows: &[Vec<Datum>],
    func: CorgiFunc,
    builtin: MirScalarExpr,
) {
    let corgi = corgi_call(&func);
    let slices: Vec<&[Datum]> = rows.iter().map(|row| &row[..]).collect();
    let mut group = c.benchmark_group(name);
    group.bench_function("builtin_rows", |b| {
        b.iter(|| {
            let arena = RowArena::new();
            for row in rows {
                black_box(builtin.eval(row, &arena).unwrap());
            }
        })
    });
    group.bench_function("corgi_rows", |b| {
        b.iter(|| {
            let arena = RowArena::new();
            for row in rows {
                black_box(corgi.eval(row, &arena).unwrap());
            }
        })
    });
    group.bench_function("corgi_batch", |b| {
        b.iter(|| {
            let arena = RowArena::new();
            black_box(func.eval_batch(&slices, &arena));
        })
    });
    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let rows: Vec<Vec<Datum>> = (0..ROWS)
        .map(|i| vec![Datum::Int64(i), Datum::Int64(-3 * i)])
        .collect();
    let func = CorgiFunc::new(
        "(input.0, input.1) add_i64".into(),
        vec![SqlScalarType::Int64, SqlScalarType::Int64],
        SqlScalarType::Int64,
    )
    .unwrap();
    let builtin = MirScalarExpr::column(0).call_binary(MirScalarExpr::column(1), AddInt64);
    bench(c, "corgi_add_10k", &rows, func, builtin);
}

fn bench_split(c: &mut Criterion) {
    let text: Vec<String> = (0..ROWS)
        .map(|i| format!("{i},{},{},{}", i * 7, i % 13, i / 3))
        .collect();
    let rows: Vec<Vec<Datum>> = text.iter().map(|s| vec![Datum::String(s)]).collect();
    let func = CorgiFunc::new(
        "input split \",\"".into(),
        vec![SqlScalarType::String],
        SqlScalarType::List {
            element_type: Box::new(SqlScalarType::String),
            custom_id: None,
        },
    )
    .unwrap();
    // The built-in `string_to_array(#0, ',')`.
    let builtin = MirScalarExpr::CallVariadic {
        func: mz_expr::func::variadic::StringToArray.into(),
        exprs: vec![
            MirScalarExpr::column(0),
            MirScalarExpr::literal_ok(Datum::String(","), mz_repr::ReprScalarType::String),
        ],
    };
    bench(c, "corgi_split_10k", &rows, func, builtin);
}

criterion_group!(benches, bench_add, bench_split);
criterion_main!(benches);
