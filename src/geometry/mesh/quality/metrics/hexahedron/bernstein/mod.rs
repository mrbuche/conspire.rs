#[cfg(test)]
mod test;

use crate::{geometry::Coordinates, math::Scalar};

const NODES: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

const SAMPLES: [Scalar; 3] = [0.0, 0.5, 1.0];

fn determinant(element: &[usize], coordinates: &Coordinates<3>, at: [Scalar; 3]) -> Scalar {
    let mut columns = [[0.0; 3]; 3];
    NODES.iter().enumerate().for_each(|(node, exponents)| {
        let point = &coordinates[element[node]];
        let value: [Scalar; 3] = std::array::from_fn(|d| {
            if exponents[d] == 1 {
                at[d]
            } else {
                1.0 - at[d]
            }
        });
        let slope: [Scalar; 3] =
            std::array::from_fn(|d| if exponents[d] == 1 { 1.0 } else { -1.0 });
        (0..3).for_each(|d| {
            let weight = slope[d] * value[(d + 1) % 3] * value[(d + 2) % 3];
            (0..3).for_each(|component| columns[d][component] += weight * point[component])
        })
    });
    let [a, b, c] = columns;
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn lift([low, middle, high]: [Scalar; 3]) -> [Scalar; 3] {
    [low, 0.5 * (4.0 * middle - low - high), high]
}

/// The coefficients of the Jacobian determinant of the trilinear map in the
/// Bernstein basis, which is tri-quadratic and so has twenty seven of them.
///
/// Bernstein basis functions are non-negative, so coefficients that are all
/// positive certify that the determinant is positive everywhere in the
/// element rather than merely at the points it is sampled at. The converse
/// does not hold: a negative coefficient is not proof of an inversion.
pub(crate) fn coefficients(element: &[usize], coordinates: &Coordinates<3>) -> [Scalar; 27] {
    let mut values = [[[0.0; 3]; 3]; 3];
    (0..3).for_each(|k| {
        (0..3).for_each(|j| {
            (0..3).for_each(|i| {
                values[k][j][i] =
                    determinant(element, coordinates, [SAMPLES[i], SAMPLES[j], SAMPLES[k]])
            })
        })
    });
    (0..3).for_each(|k| (0..3).for_each(|j| values[k][j] = lift(values[k][j])));
    (0..3).for_each(|k| {
        (0..3).for_each(|i| {
            let lifted = lift(std::array::from_fn(|j| values[k][j][i]));
            (0..3).for_each(|j| values[k][j][i] = lifted[j])
        })
    });
    (0..3).for_each(|j| {
        (0..3).for_each(|i| {
            let lifted = lift(std::array::from_fn(|k| values[k][j][i]));
            (0..3).for_each(|k| values[k][j][i] = lifted[k])
        })
    });
    std::array::from_fn(|index| values[index / 9][index / 3 % 3][index % 3])
}

/// Whether the element is certified to have a positive Jacobian throughout.
pub(crate) fn certifies(element: &[usize], coordinates: &Coordinates<3>) -> bool {
    coefficients(element, coordinates)
        .iter()
        .all(|&coefficient| coefficient > 0.0)
}

/// The least Bernstein coefficient, relative to the greatest, so that the
/// measure does not scale with the element.
pub(crate) fn margin(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    let coefficients = coefficients(element, coordinates);
    let maximum = coefficients.iter().cloned().fold(0.0, Scalar::max);
    let minimum = coefficients
        .iter()
        .cloned()
        .fold(Scalar::INFINITY, Scalar::min);
    if maximum > 0.0 {
        minimum / maximum
    } else {
        minimum
    }
}

pub(crate) fn sampled_minimum(
    element: &[usize],
    coordinates: &Coordinates<3>,
    divisions: usize,
) -> Scalar {
    let mut minimum = Scalar::INFINITY;
    (0..=divisions).for_each(|k| {
        (0..=divisions).for_each(|j| {
            (0..=divisions).for_each(|i| {
                let at = [i, j, k].map(|index| index as Scalar / divisions as Scalar);
                minimum = minimum.min(determinant(element, coordinates, at))
            })
        })
    });
    minimum
}
