#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{differential::jet::vertex_jets, tessellation::features::crease_nodes},
    },
    math::{Quantity, Scalar, Tensor},
    units::{Length, ReciprocalLength},
};

const D: usize = 3;
const N: usize = 3;

/// Choices when a feature is sharper than the tolerance.
#[derive(Clone, Copy)]
pub(crate) enum Unresolved {
    /// The smallest edge that is permissible.
    Minimum,
    /// The feature radius, saturating at sqrt(3)/curvature.
    Radius,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sizing_field(
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<D>,
    tolerance: Quantity<Length>,
    minimum: Quantity<Length>,
    maximum: Quantity<Length>,
    gradation: Scalar,
    unresolved: Unresolved,
) -> Vec<Quantity<Length>> {
    let discarded = crease_nodes(connectivity, coordinates);
    let mut field: Vec<Quantity<Length>> = vertex_jets(connectivity, coordinates, &discarded)
        .into_iter()
        .map(|jet| {
            jet.map_or(maximum, |jet| {
                dunyach_length(
                    jet.max_abs_curvature(),
                    tolerance,
                    minimum,
                    maximum,
                    unresolved,
                )
            })
        })
        .collect();
    graduate(&mut field, connectivity, coordinates, gradation);
    field
}

fn dunyach_length(
    curvature: Quantity<ReciprocalLength>,
    tolerance: Quantity<Length>,
    minimum: Quantity<Length>,
    maximum: Quantity<Length>,
    unresolved: Unresolved,
) -> Quantity<Length> {
    if curvature <= Quantity::default() {
        return maximum;
    }
    let epsilon = match unresolved {
        Unresolved::Minimum => tolerance,
        Unresolved::Radius => tolerance.min(1.0 / curvature),
    };
    let argument = epsilon * 6.0 / curvature - epsilon * epsilon * 3.0;
    let length = if argument > Quantity::default() {
        Quantity::new(argument.value().sqrt())
    } else {
        minimum
    };
    length.max(minimum).min(maximum)
}

fn graduate(
    field: &mut [Quantity<Length>],
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<D>,
    gradation: Scalar,
) {
    let mut changed = true;
    while changed {
        changed = false;
        for &[a, b, c] in connectivity {
            for (i, j) in [(a, b), (b, c), (c, a)] {
                let slope = (&coordinates[j] - &coordinates[i]).norm() * gradation;
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
