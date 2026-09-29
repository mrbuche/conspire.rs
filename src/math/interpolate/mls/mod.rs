#[cfg(test)]
mod test;

use crate::math::{SquareMatrix, SquareMatrixError, Vector};

fn extend<const D: usize>(
    axis: usize,
    degree: usize,
    current: [usize; D],
    out: &mut Vec<[usize; D]>,
) {
    if axis == D {
        out.push(current);
        return;
    }
    let used: usize = current.iter().sum();
    for power in 0..=degree - used {
        let mut next = current;
        next[axis] = power;
        extend(axis + 1, degree, next, out);
    }
}

fn exponents<const D: usize>(degree: usize) -> Vec<[usize; D]> {
    let mut out = Vec::new();
    extend(0, degree, [0; D], &mut out);
    out.sort_by_key(|exponent| exponent.iter().sum::<usize>());
    out
}

fn monomials<const D: usize>(exponents: &[[usize; D]], x: &[f64; D]) -> Vec<f64> {
    exponents
        .iter()
        .map(|exponent| {
            exponent
                .iter()
                .zip(x)
                .map(|(&power, &xk)| xk.powi(power as i32))
                .product()
        })
        .collect()
}

/// The reproducing basis functions of moving least squares, evaluated at a point.
///
/// Given the centers of the basis functions and the value of each one's weight
/// function at the point, returns the value of each basis function at the point.
/// The basis functions reproduce polynomials up to the given degree, meaning
/// the sum of each polynomial at the centers, weighted by the basis function
/// values, is the polynomial at the point. In particular the values sum to one.
pub fn moving_least_squares<const D: usize>(
    point: [f64; D],
    centers: &[[f64; D]],
    weights: &[f64],
    degree: usize,
) -> Result<Vec<f64>, SquareMatrixError> {
    assert_eq!(centers.len(), weights.len(), "Each center needs a weight.");
    let top = weights.iter().copied().fold(0.0, f64::max);
    if top <= 0.0 {
        return Err(SquareMatrixError::Singular);
    }
    let shifted: Vec<[f64; D]> = centers
        .iter()
        .map(|center| std::array::from_fn(|k| center[k] - point[k]))
        .collect();
    let scale = shifted
        .iter()
        .map(|x| x.iter().map(|xk| xk * xk).sum::<f64>().sqrt())
        .fold(0.0, f64::max);
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let exponents = exponents::<D>(degree);
    let q = exponents.len();
    let basis: Vec<Vec<f64>> = shifted
        .iter()
        .map(|x| monomials(&exponents, &std::array::from_fn(|k| x[k] / scale)))
        .collect();
    let normalized: Vec<f64> = weights.iter().map(|w| w / top).collect();
    let mut moment = SquareMatrix::zero(q);
    basis.iter().zip(&normalized).for_each(|(g, &w)| {
        for r in 0..q {
            for c in 0..q {
                moment[r][c] += w * g[r] * g[c];
            }
        }
    });
    let mut first = Vector::zero(q);
    first[0] = 1.0;
    let coefficients = moment.solve_lu(&first)?;
    Ok(basis
        .iter()
        .zip(&normalized)
        .map(|(g, &w)| w * (0..q).map(|r| coefficients[r] * g[r]).sum::<f64>())
        .collect())
}

/// The quartic weight function of a distance normalized by the support radius,
/// which is one at the center and vanishes smoothly at and beyond the radius.
pub fn quartic_weight(normalized_distance: f64) -> f64 {
    if normalized_distance >= 1.0 {
        0.0
    } else {
        (1.0 - normalized_distance * normalized_distance).powi(2)
    }
}
