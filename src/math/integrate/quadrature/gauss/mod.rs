#[cfg(test)]
mod test;

use crate::math::Scalar;
use std::f64::consts::PI;

/// Gauss-Legendre nodes and weights on $`[-1, 1]`$ with weight function $`w(x) = 1`$.
///
/// Exact for polynomials up to degree $`2n - 1`$.
///
/// ```math
/// \int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i\, f(x_i)
/// ```
pub fn gauss_legendre(n: usize) -> (Vec<Scalar>, Vec<Scalar>) {
    assert!(n > 0);
    let diagonal = vec![0.0; n];
    let off_diagonal: Vec<Scalar> = (1..n)
        .map(|k| {
            let k = k as Scalar;
            k / (4.0 * k * k - 1.0).sqrt()
        })
        .collect();
    golub_welsch(diagonal, off_diagonal, 2.0)
}

/// Generalized Gauss-Laguerre nodes and weights on $`[0, \infty)`$ with weight
/// function $`w(x) = x^{\alpha} e^{-x}`$, $`\alpha > -1`$.
///
/// Exact for polynomials (in $`x`$, against that weight) up to degree $`2n - 1`$.
///
/// ```math
/// \int_{0}^{\infty} x^{\alpha} e^{-x} f(x)\,dx \approx \sum_{i=1}^{n} w_i\, f(x_i)
/// ```
pub fn gauss_laguerre(n: usize, alpha: Scalar) -> (Vec<Scalar>, Vec<Scalar>) {
    assert!(n > 0);
    assert!(alpha > -1.0);
    let diagonal: Vec<Scalar> = (0..n).map(|k| 2.0 * k as Scalar + alpha + 1.0).collect();
    let off_diagonal: Vec<Scalar> = (1..n)
        .map(|k| {
            let k = k as Scalar;
            (k * (k + alpha)).sqrt()
        })
        .collect();
    golub_welsch(diagonal, off_diagonal, gamma(alpha + 1.0))
}

/// Gauss-Hermite nodes and weights on $`(-\infty, \infty)`$ with weight
/// function $`w(x) = e^{-x^2}`$.
///
/// Exact for polynomials up to degree $`2n - 1`$.
///
/// ```math
/// \int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx \approx \sum_{i=1}^{n} w_i\, f(x_i)
/// ```
pub fn gauss_hermite(n: usize) -> (Vec<Scalar>, Vec<Scalar>) {
    assert!(n > 0);
    let diagonal = vec![0.0; n];
    let off_diagonal: Vec<Scalar> = (1..n).map(|k| (k as Scalar / 2.0).sqrt()).collect();
    golub_welsch(diagonal, off_diagonal, PI.sqrt())
}

/// Golub-Welsch: given the symmetric tridiagonal Jacobi matrix (its `diagonal`
/// and `off_diagonal`) of a family of orthogonal polynomials and the zeroth
/// moment `mu_0` of their weight function, return the Gauss nodes (eigenvalues,
/// ascending) and weights `mu_0 * v_i0^2`.
fn golub_welsch(
    diagonal: Vec<Scalar>,
    off_diagonal: Vec<Scalar>,
    mu_0: Scalar,
) -> (Vec<Scalar>, Vec<Scalar>) {
    let n = diagonal.len();
    debug_assert_eq!(off_diagonal.len(), n.saturating_sub(1));
    let mut jacobi = vec![vec![0.0; n]; n];
    for (i, row) in jacobi.iter_mut().enumerate() {
        row[i] = diagonal[i];
    }
    for (k, &b) in off_diagonal.iter().enumerate() {
        jacobi[k][k + 1] = b;
        jacobi[k + 1][k] = b;
    }
    let (eigenvalues, first_components) = jacobi_symmetric_first_row(jacobi);
    let mut pairs: Vec<(Scalar, Scalar)> = eigenvalues
        .into_iter()
        .zip(first_components)
        .map(|(node, v0)| (node, mu_0 * v0 * v0))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    pairs.into_iter().unzip()
}

/// Cyclic Jacobi eigenvalue iteration for a symmetric matrix, tracking only the
/// first row of the eigenvector matrix (all Golub-Welsch needs). `n` is small
/// (a fixed quadrature order) and this runs once per (rule, n).
fn jacobi_symmetric_first_row(mut a: Vec<Vec<Scalar>>) -> (Vec<Scalar>, Vec<Scalar>) {
    let n = a.len();
    let mut v0 = vec![0.0; n];
    v0[0] = 1.0;
    for _ in 0..100 {
        let off_norm: Scalar = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off_norm <= 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (rp, rq) = (row[p], row[q]);
                    row[p] = c * rp - s * rq;
                    row[q] = s * rp + c * rq;
                }
                let (lower, upper) = a.split_at_mut(q); // p < q
                for (entry_p, entry_q) in lower[p].iter_mut().zip(upper[0].iter_mut()) {
                    let (pk, qk) = (*entry_p, *entry_q);
                    *entry_p = c * pk - s * qk;
                    *entry_q = s * pk + c * qk;
                }
                let (v0p, v0q) = (v0[p], v0[q]);
                v0[p] = c * v0p - s * v0q;
                v0[q] = s * v0p + c * v0q;
            }
        }
    }
    let eigenvalues = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v0)
}

/// The gamma function (Lanczos approximation, $`g = 7`$), for the zeroth moment
/// of the generalized Laguerre weight.
fn gamma(x: Scalar) -> Scalar {
    const G: Scalar = 7.0;
    const COEFFICIENTS: [Scalar; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let t = x + G + 0.5;
        let series = COEFFICIENTS
            .iter()
            .enumerate()
            .skip(1)
            .fold(COEFFICIENTS[0], |acc, (i, &c)| acc + c / (x + i as Scalar));
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * series
    }
}
