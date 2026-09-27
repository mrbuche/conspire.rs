#[cfg(test)]
mod test;

use std::array::from_fn;

/// How many of a body's six rigid-body modes are removed by holding the given
/// degrees of freedom fixed.
///
/// `positions` are the body's nodes and `constrained` its fixed degrees of
/// freedom, numbered node by node. A rigid motion moves node `x` by
/// `t + w x r`, with `r` measured from the centroid, so each fixed component
/// is one row of a matrix over the six unknowns `(t, w)`, and the modes removed
/// are its rank.
pub(crate) fn removed_modes(positions: &[[f64; 3]], constrained: &[usize]) -> usize {
    const D: usize = 3;
    if positions.is_empty() {
        return 0;
    }
    let count = positions.len() as f64;
    let centroid: [f64; D] = std::array::from_fn(|axis| {
        positions.iter().map(|position| position[axis]).sum::<f64>() / count
    });
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
    let mut columns = vec![vec![0.0; constrained.len()]; 2 * D];
    constrained.iter().enumerate().for_each(|(row, &dof)| {
        let (node, component) = (dof / D, dof % D);
        let r: [f64; D] = from_fn(|axis| (positions[node][axis] - centroid[axis]) / scale);
        columns[component][row] = 1.0;
        let (first, second) = ((component + 1) % D, (component + 2) % D);
        columns[D + first][row] = r[second];
        columns[D + second][row] = -r[first];
    });
    rank(columns)
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|entry| entry * entry).sum::<f64>().sqrt()
}

/// The number of independent columns, by Gram-Schmidt with reorthogonalization.
fn rank(columns: Vec<Vec<f64>>) -> usize {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    columns.into_iter().for_each(|mut column| {
        let before = norm(&column);
        if before == 0.0 {
            return;
        }
        for _ in 0..2 {
            basis.iter().for_each(|direction| {
                let projection: f64 = direction.iter().zip(&column).map(|(a, b)| a * b).sum();
                column
                    .iter_mut()
                    .zip(direction)
                    .for_each(|(entry, part)| *entry -= projection * part)
            })
        }
        let after = norm(&column);
        if after > 1e-8 * before {
            column.iter_mut().for_each(|entry| *entry /= after);
            basis.push(column)
        }
    });
    basis.len()
}
