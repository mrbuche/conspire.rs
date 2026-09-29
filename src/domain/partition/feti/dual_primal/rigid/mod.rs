#[cfg(test)]
mod test;

use crate::math::{Tensor, Vector};
use std::array::from_fn;

pub(crate) fn removed_modes<const D: usize>(
    positions: &[[f64; D]],
    constrained: &[usize],
) -> usize {
    if positions.is_empty() {
        return 0;
    }
    let count = positions.len() as f64;
    let centroid: [f64; D] =
        from_fn(|axis| positions.iter().map(|position| position[axis]).sum::<f64>() / count);
    let scale = positions
        .iter()
        .map(|position| {
            (0..D)
                .map(|axis| (position[axis] - centroid[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let num_rotations = D * (D - 1) / 2;
    let mut columns = vec![Vector::zero(constrained.len()); D + num_rotations];
    constrained.iter().enumerate().for_each(|(row, &dof)| {
        let (node, component) = (dof / D, dof % D);
        let r: [f64; D] = from_fn(|axis| (positions[node][axis] - centroid[axis]) / scale);
        columns[component][row] = 1.0;
        let mut pair = 0;
        (0..D).for_each(|i| {
            ((i + 1)..D).for_each(|j| {
                if component == i {
                    columns[D + pair][row] = r[j];
                } else if component == j {
                    columns[D + pair][row] = -r[i];
                }
                pair += 1;
            })
        });
    });
    rank(columns)
}

fn rank(columns: Vec<Vector>) -> usize {
    let mut basis = Vec::<Vector>::new();
    columns.into_iter().for_each(|mut column| {
        let before = column.norm().value();
        if before == 0.0 {
            return;
        }
        for _ in 0..2 {
            basis.iter().for_each(|direction| {
                let projection = direction.full_contraction(&column);
                column -= &(direction * projection);
            })
        }
        let after = column.norm().value();
        if after > 1e-8 * before {
            column /= after;
            basis.push(column)
        }
    });
    basis.len()
}

const KERNEL_TOLERANCE: f64 = 1e-8;

fn rigid_modes<const D: usize>(positions: &[[f64; D]]) -> Vec<Vector> {
    let count = positions.len() as f64;
    let centroid: [f64; D] =
        from_fn(|axis| positions.iter().map(|position| position[axis]).sum::<f64>() / count);
    let scale = positions
        .iter()
        .map(|position| {
            (0..D)
                .map(|axis| (position[axis] - centroid[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let mut modes = vec![Vector::zero(D * positions.len()); D + D * (D - 1) / 2];
    positions.iter().enumerate().for_each(|(node, position)| {
        let r: [f64; D] = from_fn(|axis| (position[axis] - centroid[axis]) / scale);
        (0..D).for_each(|component| {
            modes[component][D * node + component] = 1.0;
        });
        let mut pair = 0;
        (0..D).for_each(|i| {
            ((i + 1)..D).for_each(|j| {
                modes[D + pair][D * node + i] = r[j];
                modes[D + pair][D * node + j] = -r[i];
                pair += 1;
            })
        });
    });
    modes
}

/// The rigid-body modes a subdomain still has once `constrained` DOFs are
/// held at zero, as vectors over all its local DOFs. These span the kernel of
/// its stiffness, which is what makes a floating subdomain singular.
pub(crate) fn kernel<const D: usize>(positions: &[[f64; D]], constrained: &[usize]) -> Vec<Vector> {
    if positions.is_empty() {
        return Vec::new();
    }
    let modes = rigid_modes(positions);
    let mut basis = Vec::<(Vector, Vector)>::new();
    let mut kernel = Vec::new();
    (0..modes.len()).for_each(|mode| {
        let mut residual: Vector = constrained.iter().map(|&dof| modes[mode][dof]).collect();
        let mut combination = Vector::zero(modes.len());
        combination[mode] = 1.0;
        let before = residual.norm().value();
        for _ in 0..2 {
            basis.iter().for_each(|(direction, direction_combination)| {
                let projection = direction.full_contraction(&residual);
                residual -= &(direction * projection);
                combination -= &(direction_combination * projection);
            })
        }
        let after = residual.norm().value();
        if before == 0.0 || after <= KERNEL_TOLERANCE * before {
            let mut vector = Vector::zero(D * positions.len());
            (0..modes.len()).for_each(|other| {
                (0..vector.len())
                    .for_each(|dof| vector[dof] += combination[other] * modes[other][dof])
            });
            kernel.push(vector);
        } else {
            residual /= after;
            combination /= after;
            basis.push((residual, combination));
        }
    });
    kernel
}

/// Picks, from the DOFs `free`, as many as there are kernel vectors, such
/// that pinning them leaves the stiffness non-singular, by pivoting on the
/// row of largest norm and orthogonalizing the rest against it. Returns
/// positions within `free`, in ascending order.
pub(crate) fn kernel_pins(kernel: &[Vector], free: &[usize]) -> Vec<usize> {
    let mut rows: Vec<Vec<f64>> = free
        .iter()
        .map(|&dof| kernel.iter().map(|vector| vector[dof]).collect())
        .collect();
    let mut chosen = Vec::with_capacity(kernel.len());
    for _ in 0..kernel.len() {
        let norms: Vec<f64> = rows
            .iter()
            .map(|row| row.iter().map(|entry| entry * entry).sum())
            .collect();
        let Some((pivot, &largest)) = norms
            .iter()
            .enumerate()
            .filter(|(position, _)| !chosen.contains(position))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
        else {
            break;
        };
        if largest == 0.0 {
            break;
        }
        let direction: Vec<f64> = rows[pivot]
            .iter()
            .map(|entry| entry / largest.sqrt())
            .collect();
        rows.iter_mut().for_each(|row| {
            let projection: f64 = row.iter().zip(&direction).map(|(a, b)| a * b).sum();
            row.iter_mut()
                .zip(&direction)
                .for_each(|(entry, unit)| *entry -= projection * unit);
        });
        chosen.push(pivot);
    }
    chosen.sort_unstable();
    chosen
}
