// Copyright Materialize, Inc. and contributors. All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! Scalar functions written in corgi, a columnar term-graph language.
//!
//! A [`CorgiFunc`] is a corgi program together with the SQL types of its
//! arguments and of its result. The program sees its arguments as one column:
//! the argument's column when there is one, the product of the arguments'
//! columns when there are several, and a unit column when there are none. It
//! must produce a column of its return type's shape, which is checked when the
//! function is constructed.
//!
//! # Encoding
//!
//! A corgi leaf is unsigned bits, and does not record what the bits mean. The
//! encoding here is corgi's own idiom, at one width:
//!
//! * `bool` is a `U64` mask, zero for false. On output any nonzero is true.
//! * Signed integers are `U64` in corgi's order-preserving signed encoding
//!   ([`corgi::enc_i64`]), which its `*_i64` arithmetic and its comparisons read.
//! * Unsigned integers are `U64`, as themselves.
//! * Floats are `U64` in corgi's total-order float encoding, which its `*_f64`
//!   arithmetic and its comparisons read.
//! * `text` and `bytea` are `List<U8>`, and a list is a `List` of its elements.
//!
//! Narrower types widen to 64 bits on input and are range checked on output.
//! A program is responsible for producing the encoding its return type
//! declares: corgi's `len` produces an unsigned count, for example, which is a
//! `uint8` and not an `int8` until `signed` converts it.
//!
//! # Nulls and errors
//!
//! A null argument produces a null result without running the program. A null
//! list element is an error. A program whose result is `Sum{T | ()}`, which is
//! what corgi's `try` produces, yields null for rows in the second lane. A
//! program with a fallible stage it does not `try` is partial, and its failing
//! rows are evaluation errors.
//!
//! # Evaluation
//!
//! Evaluation is columnar. [`CorgiFunc::eval_batch`] encodes any number of rows
//! as one column, runs the program once, and decodes the result column. The
//! per-row evaluation of a `MirScalarExpr` is the one-row case, which pays
//! corgi's whole per-run cost for a single row. Handing whole batches to
//! [`CorgiFunc::eval_batch`] is the caller's concern.

use std::fmt;
use std::sync::{Arc, OnceLock};

use corgi::{Bounds, Program, Shape, Value, dec_i64, enc_i64};
use itertools::Itertools;
use mz_ore::cast::CastLossy;
use mz_repr::{Datum, RowArena, SqlColumnType, SqlScalarType};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::scalar::func::variadic::LazyVariadicFunc;
use crate::{Eval, EvalError};

/// A scalar function whose body is a corgi program.
///
/// Equality, order and hashing see the program text and the declared types.
/// The compiled program is a cache, shared by clones and rebuilt after
/// deserialization.
#[derive(Clone, Serialize, Deserialize)]
pub struct CorgiFunc {
    /// The program, in corgi's ML surface syntax.
    pub program: String,
    /// The types of the arguments, which fix the program's input shape.
    pub arg_types: Vec<SqlScalarType>,
    /// The type of the result.
    pub return_type: SqlScalarType,
    #[serde(skip)]
    compiled: CompiledCache,
}

/// The compiled program, shared by clones.
///
/// NOTE: corgi's graphs can hold host kernels, which are `dyn` objects and so
/// not `RefUnwindSafe`, while callers evaluate expressions inside
/// `catch_unwind` (the webhook validator does). Asserting unwind safety is
/// sound here: programs compiled from source hold no host kernels, and a
/// `OnceLock` is either written completely or not at all.
#[derive(Clone, Default)]
struct CompiledCache(Arc<OnceLock<Result<Compiled, String>>>);

impl std::panic::RefUnwindSafe for CompiledCache {}
impl std::panic::UnwindSafe for CompiledCache {}

/// A program, compiled and checked against the declared types.
struct Compiled {
    program: Program,
    /// The shape of each argument's column.
    arg_shapes: Vec<Shape>,
    /// Whether the program has a fallible stage it does not `try`. Its result
    /// is then `Sum{T | ()}`, whose second lane is the rows that failed.
    partial: bool,
    /// Whether the program's (successful) result is `Sum{T | ()}` rather than
    /// `T`. Its second lane is the null rows.
    nullable: bool,
}

impl CorgiFunc {
    /// Compiles `program`, and checks that on arguments of `arg_types` it
    /// produces a result of `return_type`.
    pub fn new(
        program: String,
        arg_types: Vec<SqlScalarType>,
        return_type: SqlScalarType,
    ) -> Result<Self, String> {
        let func = CorgiFunc {
            program,
            arg_types,
            return_type,
            compiled: Default::default(),
        };
        func.compiled()?;
        Ok(func)
    }

    fn compiled(&self) -> Result<&Compiled, String> {
        self.compiled
            .0
            .get_or_init(|| self.compile())
            .as_ref()
            .map_err(Clone::clone)
    }

    fn compile(&self) -> Result<Compiled, String> {
        let program = Program::compile_ml(&self.program)?;
        let arg_shapes = self
            .arg_types
            .iter()
            .map(shape_of)
            .collect::<Result<Vec<_>, _>>()?;
        let input = match &arg_shapes[..] {
            [] => Shape::Unit,
            [shape] => shape.clone(),
            shapes => Shape::Prod(shapes.to_vec()),
        };
        let expected = shape_of(&self.return_type)?;
        let partial = !program.is_total();
        let mut result = program.shape(&input)?;
        if partial {
            result = option_of(&result).ok_or_else(|| {
                format!("a partial program's result should be {{T | ()}}, but it is {result}")
            })?;
        }
        let nullable = if result == expected {
            false
        } else if option_of(&result).as_ref() == Some(&expected) {
            true
        } else {
            return Err(format!(
                "the program returns {result}, but {} is {expected}",
                type_name(&self.return_type)
            ));
        };
        Ok(Compiled {
            program,
            arg_shapes,
            partial,
            nullable,
        })
    }

    /// Evaluates the function on many rows, running the program once.
    ///
    /// Each element of `rows` holds one row's arguments. A row with a null
    /// argument is null, and is not shown to the program.
    pub fn eval_batch<'a>(
        &self,
        rows: &[&[Datum<'a>]],
        temp_storage: &'a RowArena,
    ) -> Vec<Result<Datum<'a>, EvalError>> {
        let mut results = vec![Ok(Datum::Null); rows.len()];
        let compiled = match self.compiled() {
            Ok(compiled) => compiled,
            Err(err) => {
                let err = EvalError::Internal(format!("corgi: {err}").into());
                results.fill(Err(err));
                return results;
            }
        };

        // Encode the rows the program sees, remembering where each came from.
        let mut columns: Vec<_> = self.arg_types.iter().map(|_| Column::default()).collect();
        let mut shown = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            if row.iter().any(|datum| datum.is_null()) {
                continue;
            }
            if let Err(err) = row.iter().try_for_each(check_encodable) {
                results[index] = Err(err);
                continue;
            }
            for (column, datum) in columns.iter_mut().zip_eq(row.iter()) {
                column.push(*datum);
            }
            shown.push(index);
        }
        if shown.is_empty() {
            return results;
        }
        let mut columns = columns
            .into_iter()
            .zip_eq(&compiled.arg_shapes)
            .map(|(column, shape)| column.finish(shape));
        let input = match compiled.arg_shapes.len() {
            0 => Value::Unit(shown.len()),
            1 => columns.next().expect("one column"),
            _ => Value::Prod(columns.collect()),
        };

        // A panic in corgi should fail these rows, not the process.
        let output = mz_ore::panic::catch_unwind_str(std::panic::AssertUnwindSafe(|| {
            compiled.program.run_partial(input)
        }));
        let decoded = match output {
            Ok(output) if output.len() == shown.len() => decode(
                &output,
                &self.return_type,
                compiled.partial,
                compiled.nullable,
                temp_storage,
            ),
            Ok(output) => Err(EvalError::Internal(
                format!("corgi: {} results for {} rows", output.len(), shown.len()).into(),
            )),
            Err(panic) => Err(EvalError::Internal(
                format!("corgi panicked: {panic}").into(),
            )),
        };
        match decoded {
            Ok(decoded) => {
                for (index, result) in shown.into_iter().zip_eq(decoded) {
                    results[index] = result;
                }
            }
            Err(err) => {
                for index in shown {
                    results[index] = Err(err.clone());
                }
            }
        }
        results
    }
}

impl LazyVariadicFunc for CorgiFunc {
    fn eval<'a>(
        &'a self,
        datums: &[Datum<'a>],
        temp_storage: &'a RowArena,
        exprs: &'a [impl Eval],
    ) -> Result<Datum<'a>, EvalError> {
        let args = exprs
            .iter()
            .map(|expr| expr.eval(datums, temp_storage))
            .collect::<Result<Vec<_>, _>>()?;
        self.eval_batch(&[&args[..]], temp_storage)
            .pop()
            .expect("one result per row")
    }

    fn output_type(&self, input_types: &[SqlColumnType]) -> SqlColumnType {
        let nullable = input_types.iter().any(|typ| typ.nullable)
            || self.compiled().map_or(true, |compiled| compiled.nullable);
        self.return_type.clone().nullable(nullable)
    }

    fn propagates_nulls(&self) -> bool {
        true
    }

    fn introduces_nulls(&self) -> bool {
        self.compiled().map_or(true, |compiled| compiled.nullable)
    }

    fn could_error(&self) -> bool {
        // Besides partial programs, a result can be out of range for a narrow
        // type, text can be invalid UTF-8, and a list argument can hold a null.
        true
    }
}

impl CorgiFunc {
    fn key(&self) -> (&String, &Vec<SqlScalarType>, &SqlScalarType) {
        (&self.program, &self.arg_types, &self.return_type)
    }
}

impl PartialEq for CorgiFunc {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for CorgiFunc {}

impl PartialOrd for CorgiFunc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CorgiFunc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

impl std::hash::Hash for CorgiFunc {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}

impl fmt::Debug for CorgiFunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("CorgiFunc")
            .field("program", &self.program)
            .field("arg_types", &self.arg_types)
            .field("return_type", &self.return_type)
            .finish()
    }
}

impl fmt::Display for CorgiFunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "corgi[{:?}]", self.program)
    }
}

/// The corgi shape of a SQL type, or why it has none.
fn shape_of(typ: &SqlScalarType) -> Result<Shape, String> {
    match typ {
        SqlScalarType::Bool
        | SqlScalarType::Int16
        | SqlScalarType::Int32
        | SqlScalarType::Int64
        | SqlScalarType::UInt16
        | SqlScalarType::UInt32
        | SqlScalarType::UInt64
        | SqlScalarType::Float32
        | SqlScalarType::Float64 => Ok(Shape::Prim(64)),
        SqlScalarType::String | SqlScalarType::Bytes => Ok(Shape::List(Box::new(Shape::Prim(8)))),
        SqlScalarType::List { element_type, .. } => {
            Ok(Shape::List(Box::new(shape_of(element_type)?)))
        }
        _ => Err(format!("corgi functions do not support type {typ:?}")),
    }
}

/// `T`, if `shape` is `Sum{T | ()}`.
fn option_of(shape: &Shape) -> Option<Shape> {
    match shape {
        Shape::Sum(lanes) if lanes.len() == 2 && lanes[1] == Shape::Unit => Some(lanes[0].clone()),
        _ => None,
    }
}

/// Parses the name of a type a corgi function can accept or return: `bool`,
/// `int2`, `int4`, `int8`, `uint2`, `uint4`, `uint8`, `float4`, `float8`,
/// `text` or `bytea`, followed by any number of `list`s.
pub fn corgi_type_from_name(name: &str) -> Result<SqlScalarType, String> {
    let name = name.trim().to_lowercase();
    if let Some(element) = name.strip_suffix(" list") {
        return Ok(SqlScalarType::List {
            element_type: Box::new(corgi_type_from_name(element)?),
            custom_id: None,
        });
    }
    Ok(match name.as_str() {
        "bool" | "boolean" => SqlScalarType::Bool,
        "int2" | "smallint" => SqlScalarType::Int16,
        "int4" | "int" | "integer" => SqlScalarType::Int32,
        "int8" | "bigint" => SqlScalarType::Int64,
        "uint2" => SqlScalarType::UInt16,
        "uint4" => SqlScalarType::UInt32,
        "uint8" => SqlScalarType::UInt64,
        "float4" | "real" => SqlScalarType::Float32,
        "float8" | "double precision" => SqlScalarType::Float64,
        "text" => SqlScalarType::String,
        "bytea" => SqlScalarType::Bytes,
        _ => return Err(format!("corgi functions do not support type {name:?}")),
    })
}

/// The name [`corgi_type_from_name`] parses to `typ`.
fn type_name(typ: &SqlScalarType) -> String {
    let name = match typ {
        SqlScalarType::Bool => "bool",
        SqlScalarType::Int16 => "int2",
        SqlScalarType::Int32 => "int4",
        SqlScalarType::Int64 => "int8",
        SqlScalarType::UInt16 => "uint2",
        SqlScalarType::UInt32 => "uint4",
        SqlScalarType::UInt64 => "uint8",
        SqlScalarType::Float32 => "float4",
        SqlScalarType::Float64 => "float8",
        SqlScalarType::String => "text",
        SqlScalarType::Bytes => "bytea",
        SqlScalarType::List { element_type, .. } => {
            return format!("{} list", type_name(element_type));
        }
        other => return format!("{other:?}"),
    };
    name.to_string()
}

/// corgi's total-order encoding of a float, as its `to_f64` produces.
fn enc_f64(f: f64) -> u64 {
    let bits = f.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}

fn dec_f64(u: u64) -> f64 {
    f64::from_bits(if u >> 63 == 1 { u ^ (1 << 63) } else { !u })
}

/// Errors if `datum` is a list holding a null, at any depth.
fn check_encodable(datum: &Datum) -> Result<(), EvalError> {
    if let Datum::List(list) = datum {
        for element in list.iter() {
            if element.is_null() {
                return Err(EvalError::InvalidParameterValue(
                    "corgi functions do not accept lists containing nulls".into(),
                ));
            }
            check_encodable(&element)?;
        }
    }
    Ok(())
}

/// One corgi column under construction.
///
/// Which variant a column is follows from its datums, so the column needs no
/// type. It must be given datums of one type, none null, as the argument
/// types and [`check_encodable`] ensure.
#[derive(Default)]
enum Column {
    /// No datums yet.
    #[default]
    Empty,
    /// A `U64` leaf.
    Leaf(Vec<u64>),
    /// A `List<U8>` of strings or byte strings: row ends, and bytes.
    Bytes(Vec<usize>, Vec<u8>),
    /// A `List` of some column: row ends, and elements.
    List(Vec<usize>, Box<Column>),
}

impl Column {
    fn push(&mut self, datum: Datum) {
        if let Column::Empty = self {
            *self = match datum {
                Datum::String(_) | Datum::Bytes(_) => Column::Bytes(Vec::new(), Vec::new()),
                Datum::List(_) => Column::List(Vec::new(), Box::new(Column::Empty)),
                _ => Column::Leaf(Vec::new()),
            };
        }
        match (self, datum) {
            (Column::Leaf(leaf), datum) => leaf.push(match datum {
                Datum::False => 0,
                Datum::True => 1,
                Datum::Int16(x) => enc_i64(x.into()),
                Datum::Int32(x) => enc_i64(x.into()),
                Datum::Int64(x) => enc_i64(x),
                Datum::UInt16(x) => x.into(),
                Datum::UInt32(x) => x.into(),
                Datum::UInt64(x) => x,
                Datum::Float32(x) => enc_f64(x.into_inner().into()),
                Datum::Float64(x) => enc_f64(x.into_inner()),
                datum => panic!("corgi: unencodable datum {datum:?}"),
            }),
            (Column::Bytes(ends, bytes), Datum::String(s)) => {
                bytes.extend_from_slice(s.as_bytes());
                ends.push(bytes.len());
            }
            (Column::Bytes(ends, bytes), Datum::Bytes(b)) => {
                bytes.extend_from_slice(b);
                ends.push(bytes.len());
            }
            (Column::List(ends, elements), Datum::List(list)) => {
                let mut len = ends.last().copied().unwrap_or(0);
                for element in list.iter() {
                    elements.push(element);
                    len += 1;
                }
                ends.push(len);
            }
            (_, datum) => panic!("corgi: datum {datum:?} does not match its column"),
        }
    }

    /// The finished column, of shape `shape`. The shape is needed only for a
    /// column that saw no datums, the elements of lists that were all empty.
    fn finish(self, shape: &Shape) -> Value {
        match (self, shape) {
            (Column::Empty, shape) => Value::empty(shape),
            (Column::Leaf(leaf), _) => Value::u64(leaf),
            (Column::Bytes(ends, bytes), _) => {
                Value::List(Bounds::offsets(ends), Box::new(Value::u8(bytes)))
            }
            (Column::List(ends, elements), Shape::List(element)) => {
                Value::List(Bounds::offsets(ends), Box::new(elements.finish(element)))
            }
            (Column::List(..), shape) => panic!("corgi: a list column of shape {shape}"),
        }
    }
}

/// Decodes a result column into one result per row.
///
/// The outer `Err` is a result column whose shape the program's type check
/// should have ruled out, and applies to every row.
fn decode<'a>(
    column: &Value,
    typ: &SqlScalarType,
    partial: bool,
    nullable: bool,
    temp_storage: &'a RowArena,
) -> Result<Vec<Result<Datum<'a>, EvalError>>, EvalError> {
    if partial {
        let failed = Err(EvalError::InvalidParameterValue(
            "corgi program failed on this row (a fallible stage without `try`)".into(),
        ));
        decode_option(column, failed, |ok| {
            decode(ok, typ, false, nullable, temp_storage)
        })
    } else if nullable {
        decode_option(column, Ok(Datum::Null), |some| {
            decode(some, typ, false, false, temp_storage)
        })
    } else {
        decode_column(column, typ, temp_storage)
    }
}

/// Decodes a `Sum{T | ()}` column, whose first lane `decode_first` decodes and
/// whose second lane is `other`.
fn decode_option<'a>(
    column: &Value,
    other: Result<Datum<'a>, EvalError>,
    decode_first: impl FnOnce(&Value) -> Result<Vec<Result<Datum<'a>, EvalError>>, EvalError>,
) -> Result<Vec<Result<Datum<'a>, EvalError>>, EvalError> {
    let Value::Sum(tags, lanes) = column else {
        return Err(shape_error(column));
    };
    let first = decode_first(&lanes[0])?;
    Ok((0..tags.len())
        .map(|row| match tags.tag_at(row) {
            0 => first[tags.offset_at(row)].clone(),
            _ => other.clone(),
        })
        .collect())
}

fn decode_column<'a>(
    column: &Value,
    typ: &SqlScalarType,
    temp_storage: &'a RowArena,
) -> Result<Vec<Result<Datum<'a>, EvalError>>, EvalError> {
    match typ {
        SqlScalarType::String | SqlScalarType::Bytes => {
            let Value::List(bounds, bytes) = column else {
                return Err(shape_error(column));
            };
            let bytes = bytes.as_u8("corgi").map_err(internal)?;
            let mut start = 0;
            Ok(bounds
                .to_vec()
                .into_iter()
                .map(|end| {
                    let row = &bytes[start..end];
                    start = end;
                    if let SqlScalarType::Bytes = typ {
                        return Ok(Datum::Bytes(temp_storage.push_bytes(row)));
                    }
                    match std::str::from_utf8(row) {
                        Ok(s) => Ok(Datum::String(temp_storage.push_string(s.to_owned()))),
                        Err(_) => Err(EvalError::InvalidByteSequence {
                            byte_sequence: format!("{row:?}").into(),
                            encoding_name: "UTF8".into(),
                        }),
                    }
                })
                .collect())
        }
        SqlScalarType::List { element_type, .. } => {
            let Value::List(bounds, elements) = column else {
                return Err(shape_error(column));
            };
            let elements = decode_column(elements, element_type, temp_storage)?;
            let mut start = 0;
            Ok(bounds
                .to_vec()
                .into_iter()
                .map(|end| {
                    let row = &elements[start..end];
                    start = end;
                    let row = row.iter().cloned().collect::<Result<Vec<_>, _>>()?;
                    Ok(temp_storage.make_datum(|packer| packer.push_list(row)))
                })
                .collect())
        }
        _ => {
            let leaf = column.as_u64("corgi").map_err(internal)?;
            Ok(leaf.iter().map(|&u| decode_leaf(u, typ)).collect())
        }
    }
}

fn decode_leaf<'a>(u: u64, typ: &SqlScalarType) -> Result<Datum<'a>, EvalError> {
    let signed = || dec_i64(u);
    Ok(match typ {
        SqlScalarType::Bool => Datum::from(u != 0),
        SqlScalarType::Int16 => Datum::Int16(
            i16::try_from(signed())
                .map_err(|_| EvalError::Int16OutOfRange(signed().to_string().into()))?,
        ),
        SqlScalarType::Int32 => Datum::Int32(
            i32::try_from(signed())
                .map_err(|_| EvalError::Int32OutOfRange(signed().to_string().into()))?,
        ),
        SqlScalarType::Int64 => Datum::Int64(signed()),
        SqlScalarType::UInt16 => Datum::UInt16(
            u16::try_from(u).map_err(|_| EvalError::UInt16OutOfRange(u.to_string().into()))?,
        ),
        SqlScalarType::UInt32 => Datum::UInt32(
            u32::try_from(u).map_err(|_| EvalError::UInt32OutOfRange(u.to_string().into()))?,
        ),
        SqlScalarType::UInt64 => Datum::UInt64(u),
        SqlScalarType::Float32 => {
            let f = dec_f64(u);
            let narrow = f32::cast_lossy(f);
            if narrow.is_infinite() && f.is_finite() {
                return Err(EvalError::Float32OutOfRange(f.to_string().into()));
            }
            Datum::Float32(OrderedFloat(narrow))
        }
        SqlScalarType::Float64 => Datum::Float64(OrderedFloat(dec_f64(u))),
        _ => {
            return Err(EvalError::Internal(
                format!("corgi: cannot decode {typ:?}").into(),
            ));
        }
    })
}

fn shape_error(column: &Value) -> EvalError {
    EvalError::Internal(
        format!(
            "corgi: unexpected result shape {}",
            corgi::shape_of_value(column)
        )
        .into(),
    )
}

fn internal(err: String) -> EvalError {
    EvalError::Internal(format!("corgi: {err}").into())
}

#[cfg(test)]
mod tests {
    use mz_repr::{Datum, RowArena, SqlScalarType};

    use super::*;
    use crate::{MirScalarExpr, VariadicFunc};

    fn func(program: &str, args: &[&str], ret: &str) -> CorgiFunc {
        let args = args
            .iter()
            .map(|name| corgi_type_from_name(name).unwrap())
            .collect();
        CorgiFunc::new(program.into(), args, corgi_type_from_name(ret).unwrap()).unwrap()
    }

    /// A call of `func` on the leading columns.
    fn call(func: &CorgiFunc) -> MirScalarExpr {
        MirScalarExpr::CallVariadic {
            func: VariadicFunc::Corgi(func.clone()),
            exprs: (0..func.arg_types.len())
                .map(MirScalarExpr::column)
                .collect(),
        }
    }

    /// Evaluates `expr` row by row.
    fn eval_rows<'a>(
        expr: &'a MirScalarExpr,
        rows: &[Vec<Datum<'a>>],
        arena: &'a RowArena,
    ) -> Vec<Result<Datum<'a>, EvalError>> {
        rows.iter().map(|row| expr.eval(row, arena)).collect()
    }

    #[mz_ore::test]
    fn signed_arithmetic() {
        let add = func("(input.0, input.1) add_i64", &["int8", "int8"], "int8");
        let arena = RowArena::new();
        let rows = vec![
            vec![Datum::Int64(3), Datum::Int64(-5)],
            vec![Datum::Int64(-7), Datum::Int64(2)],
            vec![Datum::Int64(i64::MAX), Datum::Int64(1)],
        ];
        let expr = call(&add);
        let results = eval_rows(&expr, &rows, &arena);
        assert_eq!(results[0], Ok(Datum::Int64(-2)));
        assert_eq!(results[1], Ok(Datum::Int64(-5)));
        // corgi's integer arithmetic wraps.
        assert_eq!(results[2], Ok(Datum::Int64(i64::MIN)));

        // Narrow types widen in, and are range checked out.
        let add = func("(input.0, input.1) add_i64", &["int2", "int4"], "int2");
        let rows = vec![
            vec![Datum::Int16(-3), Datum::Int32(1)],
            vec![Datum::Int16(1), Datum::Int32(40_000)],
        ];
        let expr = call(&add);
        let results = eval_rows(&expr, &rows, &arena);
        assert_eq!(results[0], Ok(Datum::Int16(-2)));
        assert!(matches!(results[1], Err(EvalError::Int16OutOfRange(_))));
    }

    #[mz_ore::test]
    fn comparisons_and_floats() {
        // Comparisons read the signed encoding correctly, and make masks.
        let lt = func("(input.0, input.1) lt", &["int8", "int8"], "bool");
        let arena = RowArena::new();
        let rows = vec![
            vec![Datum::Int64(-3), Datum::Int64(2)],
            vec![Datum::Int64(2), Datum::Int64(-3)],
        ];
        let expr = call(&lt);
        let results = eval_rows(&expr, &rows, &arena);
        assert_eq!(results, vec![Ok(Datum::True), Ok(Datum::False)]);

        let div = func(
            "(input.0, input.1) div_f64",
            &["float8", "float8"],
            "float8",
        );
        let rows = vec![vec![Datum::from(-3.0f64), Datum::from(2.0f64)]];
        let expr = call(&div);
        assert_eq!(
            eval_rows(&expr, &rows, &arena),
            vec![Ok(Datum::from(-1.5f64))]
        );
    }

    #[mz_ore::test]
    fn text_and_lists() {
        let arena = RowArena::new();
        let rows = vec![vec![Datum::String("a,bb,ccc")], vec![Datum::String("")]];

        let count = func("input split \",\" len", &["text"], "uint8");
        let expr = call(&count);
        let results = eval_rows(&expr, &rows, &arena);
        assert_eq!(results, vec![Ok(Datum::UInt64(3)), Ok(Datum::UInt64(1))]);

        let split = func("input split \",\"", &["text"], "text list");
        let expr = call(&split);
        let results = eval_rows(&expr, &rows, &arena);
        let words: Vec<_> = results[0]
            .as_ref()
            .unwrap()
            .unwrap_list()
            .iter()
            .map(|d| d.unwrap_str())
            .collect();
        assert_eq!(words, vec!["a", "bb", "ccc"]);

        // A list in, a list out: add one to each element.
        let succ = func(
            "input map (x -> (x, x lit_i64 1) add_i64)",
            &["int8 list"],
            "int8 list",
        );
        let list = arena.make_datum(|packer| packer.push_list([Datum::Int64(-1), Datum::Int64(5)]));
        let expr = call(&succ);
        let results = eval_rows(&expr, &[vec![list]], &arena);
        let items: Vec<_> = results[0].as_ref().unwrap().unwrap_list().iter().collect();
        assert_eq!(items, vec![Datum::Int64(0), Datum::Int64(6)]);
    }

    #[mz_ore::test]
    fn nulls_and_failures() {
        let arena = RowArena::new();
        let list = |xs: &[i64]| {
            arena.make_datum(|packer| packer.push_list(xs.iter().map(|x| Datum::Int64(*x))))
        };
        let rows = vec![vec![list(&[5, 6])], vec![list(&[])], vec![Datum::Null]];

        // `head` is fallible: without `try` the program is partial, and an
        // empty list is an error.
        let head = func("input head", &["int8 list"], "int8");
        let expr = call(&head);
        let results = eval_rows(&expr, &rows, &arena);
        assert_eq!(results[0], Ok(Datum::Int64(5)));
        assert!(matches!(
            results[1],
            Err(EvalError::InvalidParameterValue(_))
        ));
        assert_eq!(results[2], Ok(Datum::Null));
        assert!(!head.introduces_nulls());

        // With `try` the program is total, and an empty list is null.
        let head = func("input head try", &["int8 list"], "int8");
        let expr = call(&head);
        let results = eval_rows(&expr, &rows, &arena);
        assert_eq!(
            results,
            vec![Ok(Datum::Int64(5)), Ok(Datum::Null), Ok(Datum::Null)]
        );
        assert!(head.introduces_nulls());
    }

    #[mz_ore::test]
    fn type_errors() {
        let int8 = SqlScalarType::Int64;
        let text = SqlScalarType::String;
        // The result shape must match the declared return type.
        let err = CorgiFunc::new("input".into(), vec![int8.clone()], text.clone()).unwrap_err();
        assert!(err.contains("returns U64"), "{err}");
        // The program must type check on its input.
        assert!(CorgiFunc::new("input split \",\"".into(), vec![int8.clone()], text).is_err());
        // And parse.
        assert!(CorgiFunc::new("input (".into(), vec![int8.clone()], int8).is_err());
    }

    #[mz_ore::test]
    fn batch_matches_rows() {
        let arena = RowArena::new();
        let least = func(
            "((input.0, input.1) lt, input.0, input.1) select",
            &["int8", "int8"],
            "int8",
        );
        let rows: Vec<Vec<Datum>> = (0..100)
            .map(|i| {
                if i % 7 == 0 {
                    vec![Datum::Int64(i), Datum::Null]
                } else {
                    vec![Datum::Int64(i - 50), Datum::Int64(50 - i)]
                }
            })
            .collect();
        let slices: Vec<&[Datum]> = rows.iter().map(|row| &row[..]).collect();
        let batch = least.eval_batch(&slices, &arena);
        let expr = call(&least);
        assert_eq!(batch, eval_rows(&expr, &rows, &arena));
        assert_eq!(batch[1], Ok(Datum::Int64(-49)));
        assert_eq!(batch[99], Ok(Datum::Int64(-49)));
    }

    #[mz_ore::test]
    fn jaro_winkler_example_matches_strsim() {
        let program = include_str!("corgi/jaro_winkler.col")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let jaro_winkler = func(&program, &["text", "text"], "float8");

        let mut pairs: Vec<(String, String)> = [
            ("MARTHA", "MARHTA"),
            ("DWAYNE", "DUANE"),
            ("DIXON", "DICKSONX"),
            ("cheeseburger", "cheese fries"),
            ("", ""),
            ("", "abc"),
            ("abc", "bca"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        // Short strings over a small alphabet, so that matches, transpositions
        // and common prefixes are all frequent.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut string = move || {
            let len = next() % 16;
            (0..len)
                .map(|_| char::from(b'a' + u8::try_from(next() % 5).unwrap()))
                .collect::<String>()
        };
        for _ in 0..2000 {
            pairs.push((string(), string()));
        }

        let arena = RowArena::new();
        let rows: Vec<Vec<Datum>> = pairs
            .iter()
            .map(|(a, b)| vec![Datum::String(a), Datum::String(b)])
            .collect();
        let slices: Vec<&[Datum]> = rows.iter().map(|row| &row[..]).collect();
        let results = jaro_winkler.eval_batch(&slices, &arena);
        for ((a, b), result) in pairs.iter().zip_eq(results) {
            let got = result.unwrap().unwrap_float64();
            let expected = strsim::jaro_winkler(a, b);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{a:?} {b:?}: {got} != {expected}"
            );
        }
    }

    #[mz_ore::test]
    fn zero_arguments() {
        let arena = RowArena::new();
        let five = func("input lit_i64 5", &[], "int8");
        let expr = call(&five);
        let results = eval_rows(&expr, &[vec![]], &arena);
        assert_eq!(results, vec![Ok(Datum::Int64(5))]);
    }
}
