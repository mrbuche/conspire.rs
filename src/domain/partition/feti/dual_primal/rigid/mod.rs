#[cfg(test)]
mod test;

use crate::math::{Tensor, Vector};
use std::array::from_fn;

/// The rigid-body modes that constrained DOFs remove, counted incrementally.
///
/// A constrained DOF is a row of the matrix of rigid modes, a translation
/// along its axis and the rotations' reach along it, so the modes removed are
/// the rank of the rows taken.
pub(crate) struct ModeCover<'a, const D: usize> {
    centroid: [f64; D],
    scale: f64,
    positions: &'a [[f64; D]],
    basis: Vec<Vec<f64>>,
}

impl<'a, const D: usize> ModeCover<'a, D> {
    pub(crate) fn new(positions: &'a [[f64; D]]) -> Self {
        let count = positions.len().max(1) as f64;
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
        Self {
            centroid,
            scale: if scale > 0.0 { scale } else { 1.0 },
            positions,
            basis: Vec::new(),
        }
    }
    pub(crate) fn rank(&self) -> usize {
        self.basis.len()
    }
    /// Forgets the DOFs taken since the cover had this rank.
    pub(crate) fn rollback(&mut self, rank: usize) {
        self.basis.truncate(rank)
    }
    fn residual(&self, dof: usize) -> (Vec<f64>, f64, f64) {
        let (node, component) = (dof / D, dof % D);
        let r: [f64; D] =
            from_fn(|axis| (self.positions[node][axis] - self.centroid[axis]) / self.scale);
        let mut row = vec![0.0; D + D * (D - 1) / 2];
        row[component] = 1.0;
        let mut pair = 0;
        (0..D).for_each(|i| {
            ((i + 1)..D).for_each(|j| {
                if component == i {
                    row[D + pair] = r[j];
                } else if component == j {
                    row[D + pair] = -r[i];
                }
                pair += 1;
            })
        });
        let before = row.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
        for _ in 0..2 {
            self.basis.iter().for_each(|direction| {
                let projection: f64 = direction.iter().zip(&row).map(|(a, b)| a * b).sum();
                row.iter_mut()
                    .zip(direction)
                    .for_each(|(entry, direction)| *entry -= direction * projection);
            })
        }
        let after = row.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
        (row, before, after)
    }
    /// How much of a constrained DOF is new to the cover, the more the
    /// further the DOF is from the centroid and from those already taken.
    ///
    /// Where a choice is to be made, the DOF with the most holds the modes
    /// most firmly, where the modes are held only weakly by DOFs that are
    /// close together, and a stress in the tangent then tips the coarse
    /// problem indefinite.
    pub(crate) fn gain(&self, dof: usize) -> f64 {
        let (_, before, after) = self.residual(dof);
        if after > 1e-8 * before { after } else { 0.0 }
    }
    /// Takes a constrained DOF, saying whether it removed another mode.
    pub(crate) fn add(&mut self, dof: usize) -> bool {
        let (mut row, before, after) = self.residual(dof);
        if after > 1e-8 * before {
            row.iter_mut().for_each(|entry| *entry /= after);
            self.basis.push(row);
            true
        } else {
            false
        }
    }
}

pub(crate) fn removed_modes<const D: usize>(
    positions: &[[f64; D]],
    constrained: &[usize],
) -> usize {
    if positions.is_empty() {
        return 0;
    }
    let mut cover = ModeCover::new(positions);
    constrained.iter().for_each(|&dof| {
        cover.add(dof);
    });
    cover.rank()
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
