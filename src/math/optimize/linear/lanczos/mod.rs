#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Style, StyledError, Tensor, Vector, assert::AssertionError,
    optimize::OptimizationError, styled_error,
};

const CHECK: usize = 4;
const TINY: Scalar = 1e-14;
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const BREAKDOWN: Scalar = 1e-8;
const BEATEN: Scalar = 1e-8;

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

struct Pair {
    theta: Scalar,
    vector: Vector,
    mass_vector: Vector,
}

impl Lanczos {
    /// The `count` eigenpairs nearest the shift.
    ///
    /// `shift_invert` applies `(K - σM)⁻¹` and `mass` applies `M`, on vectors of
    /// length `size`. The Krylov basis is kept whole with full
    /// reorthogonalization, so memory grows as `size × steps`.
    ///
    /// A single starting vector sees one direction of a repeated eigenvalue, so
    /// the pairs found are locked and the rest of the space is searched for a
    /// pair that beats the weakest of them, until none does.
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
        let mut state = SEED;
        let mut locked = self.run(&mut shift_invert, &mut mass, &[], size, count, &mut state)?;
        while locked.len() < size {
            let best = self
                .run(&mut shift_invert, &mut mass, &locked, size, 1, &mut state)?
                .remove(0);
            let weakest = (0..locked.len())
                .min_by(|&a, &b| locked[a].theta.abs().total_cmp(&locked[b].theta.abs()))
                .unwrap_or_default();
            if best.theta.abs() <= locked[weakest].theta.abs() * (1.0 + BEATEN) {
                break;
            }
            locked[weakest] = best;
        }
        let mut pairs: Vec<(Scalar, Vector)> = locked
            .into_iter()
            .map(|pair| (shift + 1.0 / pair.theta, pair.vector))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (values, vectors) = pairs.into_iter().unzip();
        Ok(Eigenpairs { values, vectors })
    }
    /// The `want` strongest pairs of the operator in the space `M`-orthogonal to the locked ones.
    fn run(
        &self,
        shift_invert: &mut impl FnMut(&Vector) -> Vector,
        mass: &mut impl FnMut(&Vector) -> Vector,
        locked: &[Pair],
        size: usize,
        want: usize,
        state: &mut u64,
    ) -> Result<Vec<Pair>, LanczosError> {
        let free = size - locked.len();
        let want = want.min(free);
        let limit = self.max_steps.min(free);
        let mut basis = Vec::<Vector>::with_capacity(limit);
        let mut mass_basis = Vec::<Vector>::with_capacity(limit);
        let mut alphas = Vec::with_capacity(limit);
        let mut betas = Vec::with_capacity(limit);
        let (start, mass_start) =
            fresh(mass, locked, &basis, &mass_basis, size, state).ok_or(LanczosError::ZeroStart)?;
        basis.push(start);
        mass_basis.push(mass_start);
        loop {
            let j = basis.len();
            let mut w = shift_invert(&mass_basis[j - 1]);
            alphas.push(&w * &mass_basis[j - 1]);
            for _ in 0..2 {
                for pair in locked {
                    let c = &w * &pair.mass_vector;
                    w.axmy(c, &pair.vector);
                }
                for (q, mq) in basis.iter().zip(&mass_basis) {
                    let c = &w * mq;
                    w.axmy(c, q);
                }
            }
            let mut mass_w = mass(&w);
            let beta = (&w * &mass_w).max(0.0).sqrt();
            let size_t = alphas.iter().fold(0.0_f64, |m, a| m.max(a.abs())).max(TINY);
            let restart = if beta <= TINY * size_t && j < free {
                fresh(mass, locked, &basis, &mass_basis, size, state)
            } else {
                None
            };
            let broke = beta <= TINY * size_t && restart.is_none();
            let finished = j == free || broke;
            if restart.is_none() && j >= want && (finished || j.is_multiple_of(CHECK) || j == limit)
            {
                let (values, vectors) = tridiagonal(&alphas, &betas);
                let mut order: Vec<usize> = (0..j).collect();
                order.sort_by(|&a, &b| values[b].abs().total_cmp(&values[a].abs()));
                let wanted = &order[..want];
                let converged = finished
                    || wanted.iter().all(|&i| {
                        (beta * vectors[j - 1][i]).abs() <= self.rel_tol * values[i].abs()
                    });
                if converged {
                    return Ok(wanted
                        .iter()
                        .map(|&i| {
                            let mut vector = Vector::zero(size);
                            let mut mass_vector = Vector::zero(size);
                            for (k, (q, mq)) in basis.iter().zip(&mass_basis).enumerate() {
                                vector.axpy(vectors[k][i], q);
                                mass_vector.axpy(vectors[k][i], mq);
                            }
                            Pair {
                                theta: values[i],
                                vector,
                                mass_vector,
                            }
                        })
                        .collect());
                }
            }
            if finished {
                return Err(LanczosError::Exhausted(j, want));
            }
            if j >= limit {
                return Err(LanczosError::MaximumStepsReached(limit));
            }
            match restart {
                Some((vector, mass_vector)) => {
                    betas.push(0.0);
                    basis.push(vector);
                    mass_basis.push(mass_vector);
                }
                None => {
                    betas.push(beta);
                    w /= beta;
                    mass_w /= beta;
                    basis.push(w);
                    mass_basis.push(mass_w);
                }
            }
        }
    }
}

/// A random vector `M`-orthonormal to the locked pairs and the basis, if any direction is left.
fn fresh(
    mass: &mut impl FnMut(&Vector) -> Vector,
    locked: &[Pair],
    basis: &[Vector],
    mass_basis: &[Vector],
    size: usize,
    state: &mut u64,
) -> Option<(Vector, Vector)> {
    let mut vector: Vector = (0..size)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state >> 11) as Scalar / (1_u64 << 53) as Scalar - 0.5
        })
        .collect();
    let before = (&vector * &mass(&vector)).sqrt();
    if before.is_nan() || before <= 0.0 {
        return None;
    }
    for _ in 0..2 {
        for pair in locked {
            let c = &vector * &pair.mass_vector;
            vector.axmy(c, &pair.vector);
        }
        for (q, mq) in basis.iter().zip(mass_basis) {
            let c = &vector * mq;
            vector.axmy(c, q);
        }
    }
    let mut mass_vector = mass(&vector);
    let norm = (&vector * &mass_vector).max(0.0).sqrt();
    if norm.is_nan() || norm <= BREAKDOWN * before {
        return None;
    }
    vector /= norm;
    mass_vector /= norm;
    Some((vector, mass_vector))
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
    /// No directions were left to search after finding this many of the wanted pairs.
    Exhausted(usize, usize),
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
            Self::Exhausted(found, wanted) => format!(
                "{h}The space was exhausted after {found} of the {wanted} wanted eigenpairs.{c}\n\
                The mass may be singular on what is left."
            ),
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
