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
