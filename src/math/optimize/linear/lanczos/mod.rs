#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Style, StyledError, Vector, assert::AssertionError, optimize::OptimizationError,
    styled_error,
};

const CHECK: usize = 4;
const TINY: Scalar = 1e-14;

/// Shift-invert Lanczos for the generalized symmetric eigenproblem `K x = λ M x`.
#[derive(Clone, Copy, Debug)]
pub struct Lanczos {
    /// Maximum size of the Krylov basis.
    pub max_steps: usize,
    /// Tolerance on Ritz residuals, relative to the Ritz value.
    pub rel_tol: Scalar,
}

impl Default for Lanczos {
    fn default() -> Self {
        Self {
            max_steps: 500,
            rel_tol: 1e-10,
        }
    }
}

/// Eigenvalues and `M`-orthonormal eigenvectors, ascending in eigenvalue.
#[derive(Clone, Debug)]
pub struct Eigenpairs {
    pub values: Vec<Scalar>,
    pub vectors: Vec<Vector>,
}

impl Lanczos {
    /// The `count` eigenpairs nearest the shift.
    ///
    /// `shift_invert` applies `(K - σM)⁻¹` and `mass` applies `M`, on vectors of
    /// length `size`. The Krylov basis is kept whole with full
    /// reorthogonalization, so memory grows as `size × steps`.
    pub fn solve(
        &self,
        mut shift_invert: impl FnMut(&Vector) -> Vector,
        mut mass: impl FnMut(&Vector) -> Vector,
        shift: Scalar,
        size: usize,
        count: usize,
    ) -> Result<Eigenpairs, LanczosError> {
        if count == 0 || count > size {
            return Err(LanczosError::InvalidCount(count, size));
        }
        let limit = self.max_steps.min(size);
        let mut basis = Vec::<Vector>::with_capacity(limit);
        let mut mass_basis = Vec::<Vector>::with_capacity(limit);
        let mut alphas = Vec::with_capacity(limit);
        let mut betas = Vec::with_capacity(limit);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut start: Vector = (0..size)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as Scalar / (1_u64 << 53) as Scalar - 0.5
            })
            .collect();
        let mut mass_start = mass(&start);
        let norm = dot(&start, &mass_start).sqrt();
        if norm.is_nan() || norm <= 0.0 {
            return Err(LanczosError::ZeroStart);
        }
        scale(&mut start, 1.0 / norm);
        scale(&mut mass_start, 1.0 / norm);
        basis.push(start);
        mass_basis.push(mass_start);
        loop {
            let j = basis.len();
            let mut w = shift_invert(&mass_basis[j - 1]);
            alphas.push(dot(&w, &mass_basis[j - 1]));
            for _ in 0..2 {
                for (q, mq) in basis.iter().zip(&mass_basis) {
                    let c = dot(&w, mq);
                    axpy(&mut w, -c, q);
                }
            }
            let mut mass_w = mass(&w);
            let beta = dot(&w, &mass_w).max(0.0).sqrt();
            let size_t = alphas.iter().fold(0.0_f64, |m, a| m.max(a.abs())).max(TINY);
            let exhausted = beta <= TINY * size_t || j == size;
            if j >= count && (exhausted || j.is_multiple_of(CHECK) || j == limit) {
                let (values, vectors) = tridiagonal(&alphas, &betas);
                let mut order: Vec<usize> = (0..j).collect();
                order.sort_by(|&a, &b| values[b].abs().total_cmp(&values[a].abs()));
                let wanted = &order[..count];
                let converged = exhausted
                    || wanted.iter().all(|&i| {
                        (beta * vectors[j - 1][i]).abs() <= self.rel_tol * values[i].abs()
                    });
                if converged {
                    let mut pairs: Vec<(Scalar, Vector)> = wanted
                        .iter()
                        .map(|&i| {
                            let mut x = Vector::zero(size);
                            for (k, q) in basis.iter().enumerate() {
                                axpy(&mut x, vectors[k][i], q);
                            }
                            (shift + 1.0 / values[i], x)
                        })
                        .collect();
                    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let (values, vectors) = pairs.into_iter().unzip();
                    return Ok(Eigenpairs { values, vectors });
                }
            }
            if j >= limit {
                return Err(LanczosError::MaximumStepsReached(limit));
            }
            betas.push(beta);
            scale(&mut w, 1.0 / beta);
            scale(&mut mass_w, 1.0 / beta);
            basis.push(w);
            mass_basis.push(mass_w);
        }
    }
}

fn dot(a: &Vector, b: &Vector) -> Scalar {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| x * y)
        .sum()
}

fn scale(a: &mut Vector, factor: Scalar) {
    a.as_mut_slice().iter_mut().for_each(|x| *x *= factor);
}

fn axpy(y: &mut Vector, factor: Scalar, x: &Vector) {
    y.as_mut_slice()
        .iter_mut()
        .zip(x.as_slice())
        .for_each(|(y, x)| *y += factor * x);
}

/// Eigen-decomposition of a symmetric tridiagonal matrix by implicit QL:
/// the eigenvalues, and `vectors[i][k]` the `i`-th component of eigenvector `k`.
fn tridiagonal(diagonal: &[Scalar], off_diagonal: &[Scalar]) -> (Vec<Scalar>, Vec<Vec<Scalar>>) {
    let n = diagonal.len();
    let mut d = diagonal.to_vec();
    let mut e = off_diagonal.to_vec();
    e.resize(n, 0.0);
    let mut z = vec![vec![0.0; n]; n];
    (0..n).for_each(|i| z[i][i] = 1.0);
    let mut f = 0.0;
    let mut tst1: Scalar = 0.0;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > Scalar::EPSILON * tst1 {
            m += 1;
        }
        if m > l {
            for _ in 0..100 {
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                (l + 2..n).for_each(|i| d[i] -= h);
                f += h;
                p = d[m];
                let (mut c, mut c2, mut c3) = (1.0, 1.0, 1.0);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0.0, 0.0);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for row in z.iter_mut() {
                        h = row[i + 1];
                        row[i + 1] = s * row[i] + c * h;
                        row[i] = c * row[i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= Scalar::EPSILON * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    (d, z)
}

/// Possible errors encountered during a Lanczos eigensolve.
pub enum LanczosError {
    InvalidCount(usize, usize),
    MaximumStepsReached(usize),
    ZeroStart,
}

impl StyledError for LanczosError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::InvalidCount(count, size) => {
                format!("{h}Cannot find {count} eigenpairs of a problem of size {size}.{c}")
            }
            Self::MaximumStepsReached(steps) => {
                format!("{h}The wanted eigenpairs did not converge in {steps} iterations.{c}")
            }
            Self::ZeroStart => format!("{h}The starting vector has no mass.{c}"),
        }
    }
}

styled_error!(LanczosError);

impl From<LanczosError> for OptimizationError {
    fn from(error: LanczosError) -> Self {
        Self::Upstream(error.to_string(), "Lanczos".to_string())
    }
}

impl From<LanczosError> for AssertionError {
    fn from(error: LanczosError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
