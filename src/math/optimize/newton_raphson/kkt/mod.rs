use super::super::{
    super::{
        HessianBlock, Jacobian, LuDecomposition, Scalar, Solution, SquareMatrix, Tensor, Vector,
        sparse::{CscMatrix, SparseSolver},
    },
    OptimizationError,
};
use std::{collections::BTreeMap, ops::Mul};

use super::NewtonRaphson;
use super::convergence::{converged, limit_decrement};

#[expect(clippy::too_many_arguments)]
pub(super) fn kkt_entry<Kuu, Kvu, Kuv, Kvv>(
    row: usize,
    column: usize,
    num_global: usize,
    num_outer: usize,
    num_local: usize,
    tangent_uu: &Kuu,
    tangent_vu: &Kvu,
    tangent_uv: &Kuv,
    tangent_vv: &Kvv,
    constraint_matrix_global: &CscMatrix,
    constraint_matrix_local: &CscMatrix,
) -> Scalar
where
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
{
    let local = num_outer + num_local;
    let row_global = row < num_global;
    let row_local = (num_outer..local).contains(&row);
    let column_global = column < num_global;
    let column_local = (num_outer..local).contains(&column);
    if row_global && column_global {
        tangent_uu.entry(row, column)
    } else if row_global && column_local {
        tangent_uv.entry(row, column - num_outer)
    } else if row_local && column_global {
        tangent_vu.entry(row - num_outer, column)
    } else if row_local && column_local {
        tangent_vv.entry(row - num_outer, column - num_outer)
    } else if row_global && (num_global..num_outer).contains(&column) {
        -constraint_matrix_global.entry(column - num_global, row)
    } else if (num_global..num_outer).contains(&row) && column_global {
        -constraint_matrix_global.entry(row - num_global, column)
    } else if row_local && column >= local {
        -constraint_matrix_local.entry(column - local, row - num_outer)
    } else if row >= local && column_local {
        -constraint_matrix_local.entry(row - local, column - num_outer)
    } else {
        0.0
    }
}

pub(super) fn kkt_block<K>(
    tangent: &K,
    constraint_matrix: &CscMatrix,
    size: usize,
    block: &mut SquareMatrix,
    offset: usize,
) where
    K: HessianBlock,
{
    tangent.fill_into_block(block, offset, offset);
    constraint_matrix.iter().for_each(|(a, j, entry)| {
        block[offset + size + a][offset + j] = -entry;
        block[offset + j][offset + size + a] = -entry;
    })
}

pub(super) fn kkt_residual<R, T>(
    residual: R,
    multipliers: &Vector,
    constraint_matrix: &CscMatrix,
    constraint_rhs: &Vector,
    variables: &T,
    chained: &mut Vector,
) where
    R: Jacobian,
    for<'a> &'a CscMatrix: Mul<&'a T, Output = Vector>,
{
    (residual - multipliers * constraint_matrix)
        .fill_into_chained(constraint_rhs - constraint_matrix * variables, chained)
}

/// Converges the local variables at fixed global ones.
///
/// The condensed strategy treats the local variables as a function of the
/// global ones, so anywhere the global variables are moved to, this is what
/// the local ones become.
#[expect(clippy::too_many_arguments)]
pub(super) fn converge_local<U, V, Rv, Kuu, Kvu, Kuv, Kvv>(
    local_solver: &NewtonRaphson,
    residual_local: &mut impl FnMut(&U, &V) -> Result<Rv, String>,
    tangents: &mut impl FnMut(&U, &V) -> Result<(Kuu, Kvu, Kuv, Kvv), String>,
    global: &U,
    local: &mut V,
    multipliers_local: &mut Vector,
    constraint_matrix_local: &CscMatrix,
    constraint_rhs_local: &Vector,
    num_local: usize,
    update_inner: &mut Vector,
    tangent_inner: &mut SquareMatrix,
    factorization: &mut LuDecomposition,
) -> Result<(), OptimizationError>
where
    Rv: Jacobian,
    Kvv: HessianBlock,
    V: Solution,
    for<'a> &'a CscMatrix: Mul<&'a V, Output = Vector>,
{
    let mut local_steps = 0;
    let mut scales = None;
    loop {
        kkt_residual(
            residual_local(global, local)?,
            multipliers_local,
            constraint_matrix_local,
            constraint_rhs_local,
            local,
            update_inner,
        );
        if converged(local_solver, update_inner, num_local, &mut scales)
            || local_steps == local_solver.max_steps
        {
            return Ok(());
        }
        local_steps += 1;
        let (_, _, _, tangent) = tangents(global, local)?;
        kkt_block(
            &tangent,
            constraint_matrix_local,
            num_local,
            tangent_inner,
            0,
        );
        tangent_inner.factorize_lu_into(factorization)?;
        let mut decrement = factorization.solve(update_inner);
        limit_decrement(local_solver, &mut [(&mut decrement, num_local)]);
        local.decrement_from_chained(multipliers_local, &decrement)
    }
}

/// Eliminates the local unknowns from a Newton step in which the local block is
/// independent square blocks, one factorization each, so that only the global system,
/// with the Schur complement $`K_{uu}-K_{uv}K_{vv}^{-1}K_{vu}`$ accumulated onto its
/// sparsity pattern, is handed to the sparse solver. The local step follows by back
/// substitution.
#[expect(clippy::too_many_arguments)]
pub(super) fn eliminate_blocks<Kuu, Kvu, Kuv, Kvv>(
    solver: &SparseSolver,
    schur: &mut CscMatrix,
    tangent_uu: &Kuu,
    tangent_vu: &Kvu,
    tangent_uv: &Kuv,
    tangent_vv: &Kvv,
    constraint_matrix_global: &CscMatrix,
    (num_global, num_outer, num_local): (usize, usize, usize),
    update_outer: &mut Vector,
    update_inner: &Vector,
    decrement_outer: &mut Vector,
    decrement_inner: &mut Vector,
) -> Result<(), OptimizationError>
where
    Kuu: HessianBlock,
    Kvu: HessianBlock,
    Kuv: HessianBlock,
    Kvv: HessianBlock,
{
    let size = tangent_vv
        .block_size()
        .expect("Eliminating the local block sparsely needs it held as independent blocks.");
    let num_blocks = num_local / size;
    let mut coupling_local = vec![Vec::new(); num_blocks];
    let mut coupling_global = vec![Vec::new(); num_blocks];
    let mut blocks = vec![0.0; num_blocks * size * size];
    let held = tangent_vu.for_each_entry(&mut |row, column, value| {
        coupling_local[row / size].push((row % size, column, value))
    }) && tangent_uv.for_each_entry(&mut |row, column, value| {
        coupling_global[column / size].push((row, column % size, value))
    }) && tangent_vv.for_each_entry(&mut |row, column, value| {
        assert_eq!(
            row / size,
            column / size,
            "The local block is not block diagonal."
        );
        blocks[(row / size) * size * size + (row % size) * size + column % size] += value
    });
    assert!(
        held,
        "Eliminating the local block sparsely needs the couplings held as sparse entries."
    );
    schur.clear();
    let mut eliminated = Vec::with_capacity(num_blocks);
    for (block, (coupling_local, coupling_global)) in coupling_local
        .iter()
        .zip(coupling_global.iter())
        .enumerate()
    {
        let offset = block * size;
        let lu = (0..size)
            .map(|i| {
                let start = block * size * size + i * size;
                Vector::from(blocks[start..start + size].to_vec())
            })
            .collect::<SquareMatrix>()
            .factorize_lu()?;
        let step = lu.solve(&Vector::from(
            update_inner.as_slice()[offset..offset + size].to_vec(),
        ));
        (0..size).for_each(|l| decrement_inner[offset + l] = step[l]);
        let mut columns = BTreeMap::new();
        coupling_local.iter().for_each(|&(row, column, value)| {
            columns.entry(column).or_insert_with(|| vec![0.0; size])[row] += value
        });
        let solved: Vec<(usize, Vector)> = columns
            .into_iter()
            .map(|(column, entries)| (column, lu.solve(&Vector::from(entries))))
            .collect();
        let mut rows = BTreeMap::new();
        coupling_global.iter().for_each(|&(row, column, value)| {
            rows.entry(row).or_insert_with(|| vec![0.0; size])[column] += value
        });
        rows.into_iter().for_each(|(row, entries)| {
            update_outer[row] -= entries
                .iter()
                .zip(step.iter())
                .map(|(entry, step_l)| entry * step_l)
                .sum::<Scalar>();
            solved.iter().for_each(|(column, solution)| {
                schur.accumulate(
                    row,
                    *column,
                    entries
                        .iter()
                        .zip(solution.iter())
                        .map(|(entry, solution_l)| entry * solution_l)
                        .sum::<Scalar>(),
                )
            })
        });
        eliminated.push(solved)
    }
    let no_local = CscMatrix::from_pattern(0, 0, Vec::new());
    let schur = &*schur;
    *decrement_outer = solver.solve(
        |i, j| {
            kkt_entry(
                i,
                j,
                num_global,
                num_outer,
                0,
                tangent_uu,
                tangent_vu,
                tangent_uv,
                tangent_vv,
                constraint_matrix_global,
                &no_local,
            ) - if i < num_global && j < num_global {
                schur.entry(i, j)
            } else {
                0.0
            }
        },
        update_outer,
    )?;
    eliminated.iter().enumerate().for_each(|(block, solved)| {
        solved.iter().for_each(|(column, solution)| {
            let decrement = decrement_outer[*column];
            (0..size).for_each(|l| decrement_inner[block * size + l] -= solution[l] * decrement)
        })
    });
    Ok(())
}
