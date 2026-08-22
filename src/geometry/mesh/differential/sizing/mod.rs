#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{differential::jet::vertex_jets, tessellation::features::crease_nodes},
    },
    math::{FxHashSet, Quantity, Scalar, Tensor},
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

/// Choices for how a crease enters the curvature fit.
#[derive(Clone, Copy)]
pub(crate) enum Creases {
    /// Fit through them, like any other vertex.
    Included,
    /// Leave them out, sizing them from the neighbors that remain.
    Discarded,
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
    creases: Creases,
) -> Vec<Quantity<Length>> {
    let discarded = match creases {
        Creases::Included => FxHashSet::default(),
        Creases::Discarded => crease_nodes(connectivity, coordinates),
    };
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
    if !discarded.is_empty() {
        seed_discarded(&mut field, connectivity, &discarded, maximum);
    }
    graduate(&mut field, connectivity, coordinates, gradation);
    field
}

fn seed_discarded(
    field: &mut [Quantity<Length>],
    connectivity: &[[usize; N]],
    discarded: &FxHashSet<usize>,
    maximum: Quantity<Length>,
) {
    let mut seeded = vec![maximum; field.len()];
    for &[a, b, c] in connectivity {
        for (i, j) in [(a, b), (b, c), (c, a), (b, a), (c, b), (a, c)] {
            if discarded.contains(&i) && !discarded.contains(&j) && field[j] < seeded[i] {
                seeded[i] = field[j]
            }
        }
    }
    discarded
        .iter()
        .for_each(|&vertex| field[vertex] = seeded[vertex]);
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
