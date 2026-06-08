#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinates, mesh::differential::jet::vertex_jets},
    math::{Scalar, Tensor},
};

const N: usize = 3;

/// Per-vertex target edge length: curvature (via [`vertex_jets`]) mapped through the Dunyach
/// sizing law, then graduated so the field is Lipschitz.
pub fn sizing_field(
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<3>,
    tolerance: Scalar,
    minimum: Scalar,
    maximum: Scalar,
    gradation: Scalar,
) -> Vec<Scalar> {
    let mut field: Vec<Scalar> = vertex_jets(connectivity, coordinates)
        .into_iter()
        .map(|jet| {
            jet.map_or(maximum, |jet| {
                dunyach_length(jet.max_abs_curvature(), tolerance, minimum, maximum)
            })
        })
        .collect();
    graduate(&mut field, connectivity, coordinates, gradation);
    field
}

/// `L = sqrt(6*e/k - 3*e^2)` (chord error `e`, curvature `k`), clamped to `[minimum, maximum]`.
fn dunyach_length(
    curvature: Scalar,
    tolerance: Scalar,
    minimum: Scalar,
    maximum: Scalar,
) -> Scalar {
    if curvature <= 0.0 {
        return maximum;
    }
    let argument = 6.0 * tolerance / curvature - 3.0 * tolerance * tolerance;
    let length = if argument > 0.0 {
        argument.sqrt()
    } else {
        minimum
    };
    length.clamp(minimum, maximum)
}

/// Enforces `|L_i - L_j| <= gradation * ||x_i - x_j||` over edges by min-propagation to a
/// fixed point, so target size never changes faster than the gradation slope.
fn graduate(
    field: &mut [Scalar],
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<3>,
    gradation: Scalar,
) {
    let mut changed = true;
    while changed {
        changed = false;
        for &[a, b, c] in connectivity {
            for (i, j) in [(a, b), (b, c), (c, a)] {
                let slope = gradation * (&coordinates[j] - &coordinates[i]).norm();
                if field[i] + slope < field[j] {
                    field[j] = field[i] + slope;
                    changed = true;
                } else if field[j] + slope < field[i] {
                    field[i] = field[j] + slope;
                    changed = true;
                }
            }
        }
    }
}
