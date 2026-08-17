#[cfg(test)]
mod test;

use super::{
    super::{
        Hessian, Scalar, Style, StyledError, Tensor, Vector, assert::AssertionError, styled_error,
    },
    OptimizationError,
};

/// What the residual is divided through by on its way to becoming a direction.
///
/// Krylov methods converge on how nearly the system behaves like a multiple of
/// the identity, which a tangent assembled from unlike quantities does not.
/// Dividing through by something of the same kind as the tangent is what brings
/// the two nearer.
#[derive(Clone, Copy, Debug, Default)]
pub enum Preconditioner {
    /// The diagonal of the tangent.
    #[default]
    Jacobi,
    /// Nothing.
    None,
}

/// Which walk through the Krylov subspace to take.
///
/// Both want the tangent symmetric. They part on what else they want of it, and
/// the one that wants less of it does more work per iteration to get by.
#[derive(Clone, Copy, Debug, Default)]
pub enum KrylovMethod {
    /// Conjugate gradients, descending the quadratic the system is the
    /// stationary point of.
    ///
    /// Needs the tangent positive definite, and says so when it is not. Where
    /// that holds it is the cheaper of the two, keeping three vectors rather
    /// than six.
    #[default]
    ConjugateGradients,
    /// The minimal residual method, shortening the residual over the subspace
    /// reached so far.
    ///
    /// A quadratic with no minimum still has a residual with a shortest length,
    /// so this asks only for symmetry and serves the systems conjugate gradients
    /// has to refuse.
    Minres,
}

/// An iterative linear solve, asking the tangent only what it does to a vector.
///
/// Nothing is factorized and nothing is assembled, so the tangent is free never
/// to exist as a matrix at all. What that costs is exactness: the solve stops
/// once the residual it leaves behind is small enough against the one it started
/// from, rather than at an answer.
///
/// Symmetry is a caller-supplied guarantee either way, and neither walk can tell
/// it has been given something else.
#[derive(Clone, Copy, Debug)]
pub struct Krylov {
    /// Maximum number of iterations.
    pub max_steps: usize,
    /// Which walk to take.
    pub method: KrylovMethod,
    /// Preconditioner.
    pub preconditioner: Preconditioner,
    /// Residual tolerance, relative to the residual started from.
    pub rel_tol: Scalar,
}

impl Default for Krylov {
    fn default() -> Self {
        Self {
            max_steps: 1_000,
            method: KrylovMethod::default(),
            preconditioner: Preconditioner::default(),
            rel_tol: 1e-10,
        }
    }
}

impl Krylov {
    /// The diagonal to divide through by, at the given positions of a tangent.
    ///
    /// The magnitude of each entry is taken rather than the entry. What divides
    /// the residual has to be positive definite for either walk to hold its
    /// footing, and the diagonal of an indefinite tangent is not; on a positive
    /// definite one, where the entries are already positive, taking magnitudes
    /// changes nothing.
    ///
    /// A zero on the diagonal is left as a one rather than divided by, a row the
    /// tangent says nothing about not being one to scale by nothing.
    fn diagonal<H>(&self, tangent: &H, positions: impl Iterator<Item = usize>) -> Option<Vector>
    where
        H: Hessian,
    {
        match self.preconditioner {
            Preconditioner::Jacobi => Some(
                positions
                    .map(|i| match tangent.entry(i, i).abs() {
                        0.0 => 1.0,
                        entry => entry,
                    })
                    .collect(),
            ),
            Preconditioner::None => None,
        }
    }
    /// Solves a system, by whichever walk was asked for.
    pub fn solve<H>(&self, tangent: &H, right_hand_side: &Vector) -> Result<Vector, KrylovError>
    where
        H: Hessian,
    {
        self.walk(
            |direction| tangent.times(direction),
            self.diagonal(tangent, 0..right_hand_side.len()),
            right_hand_side,
        )
    }
    /// Solves against only the retained positions of a tangent.
    ///
    /// The retained positions are scattered out into the whole and the product
    /// gathered back from it, rather than the tangent being addressed through
    /// the mapping one entry at a time. Whatever was left out stands at zero, so
    /// it lends nothing to the rows that were kept, and the tangent is still
    /// walked the efficient way it knows.
    pub fn solve_retained<H>(
        &self,
        tangent: &H,
        unmap: &[usize],
        size: usize,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError>
    where
        H: Hessian,
    {
        let mut whole = Vector::zero(size);
        self.walk(
            |direction| {
                whole.iter_mut().for_each(|entry| *entry = 0.0);
                unmap
                    .iter()
                    .zip(direction.iter())
                    .for_each(|(&index, entry)| whole[index] = *entry);
                let applied = tangent.times(&whole);
                unmap.iter().map(|&index| applied[index]).collect()
            },
            self.diagonal(tangent, unmap.iter().copied()),
            right_hand_side,
        )
    }
    /// Solves a system given only what it does to a vector.
    ///
    /// The tangent need never exist, so this is what a system assembled from
    /// pieces is reached through.
    pub fn solve_operator(
        &self,
        apply: impl FnMut(&Vector) -> Vector,
        diagonal: Option<Vector>,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
        self.walk(apply, diagonal, right_hand_side)
    }
    fn walk(
        &self,
        apply: impl FnMut(&Vector) -> Vector,
        diagonal: Option<Vector>,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
        match self.method {
            KrylovMethod::ConjugateGradients => self.descend(apply, diagonal, right_hand_side),
            KrylovMethod::Minres => self.minimize_residual(apply, diagonal, right_hand_side),
        }
    }
    /// Divides the residual through by the preconditioner, if there is one.
    fn divide(residual: &Vector, diagonal: Option<&Vector>) -> Vector {
        match diagonal {
            Some(diagonal) => residual
                .iter()
                .zip(diagonal.iter())
                .map(|(entry, scale)| entry / scale)
                .collect(),
            None => residual.clone(),
        }
    }
    /// Descends the quadratic the system is the stationary point of.
    ///
    /// Each direction is conjugate to the ones before it against the tangent, so
    /// the step along it is exact and never needs revisiting, which is what lets
    /// the whole solve be a walk rather than a factorization.
    ///
    /// Positive definiteness is what that rests on, and the walk says so when it
    /// turns out to be missing: a direction of nonpositive curvature is one
    /// along which the quadratic has no minimum to step to.
    fn descend(
        &self,
        mut apply: impl FnMut(&Vector) -> Vector,
        diagonal: Option<Vector>,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
        let scale = right_hand_side.norm().value();
        let mut solution = Vector::zero(right_hand_side.len());
        if scale == 0.0 {
            return Ok(solution);
        }
        let divide = |residual: &Vector| Self::divide(residual, diagonal.as_ref());
        let mut residual = right_hand_side.clone();
        let mut preconditioned = divide(&residual);
        let mut direction = preconditioned.clone();
        let mut projection = residual.full_contraction(&preconditioned);
        let mut applied;
        let mut curvature;
        let mut next;
        let mut step;
        for _ in 0..self.max_steps {
            applied = apply(&direction);
            curvature = direction.full_contraction(&applied);
            if curvature <= 0.0 {
                return Err(KrylovError::NotPositiveDefinite(curvature));
            }
            step = projection / curvature;
            solution += &direction * step;
            residual -= applied * step;
            if residual.norm().value() <= self.rel_tol * scale {
                return Ok(solution);
            }
            preconditioned = divide(&residual);
            next = residual.full_contraction(&preconditioned);
            direction *= next / projection;
            direction += &preconditioned;
            projection = next
        }
        Err(KrylovError::MaximumStepsReached(
            self.max_steps,
            residual.norm().value() / scale,
        ))
    }
    /// Shortens the residual over the subspace reached so far.
    ///
    /// The tangent is turned into a tridiagonal one over an orthonormal basis
    /// built three vectors at a time, and that small system is kept factorized
    /// by a rotation per iteration. Shortest residual is a least squares
    /// question, which has an answer whether or not the quadratic has a minimum,
    /// so nothing here asks the tangent to be definite.
    ///
    /// The rotations leave the length of the residual behind as a number, so
    /// what would otherwise be another product to measure convergence by is
    /// already in hand.
    fn minimize_residual(
        &self,
        mut apply: impl FnMut(&Vector) -> Vector,
        diagonal: Option<Vector>,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError> {
        let size = right_hand_side.len();
        let mut solution = Vector::zero(size);
        let divide = |residual: &Vector| Self::divide(residual, diagonal.as_ref());
        let mut previous = right_hand_side.clone();
        let mut current = previous.clone();
        let mut preconditioned = divide(&previous);
        let scale = previous.full_contraction(&preconditioned).max(0.0).sqrt();
        if scale == 0.0 {
            return Ok(solution);
        }
        //
        // The rotation starts as a half turn so that the first iteration takes
        // the diagonal of the tridiagonal system as it stands.
        //
        let (mut cosine, mut sine) = (-1.0, 0.0);
        let mut length = scale;
        let mut off_diagonal = scale;
        let (mut previous_off, mut carried, mut trailing) = (0.0, 0.0, 0.0 as Scalar);
        let mut direction = Vector::zero(size);
        let mut older;
        let mut old = Vector::zero(size);
        let mut basis;
        for step in 0..self.max_steps {
            basis = &preconditioned * off_diagonal.recip();
            preconditioned = apply(&basis);
            if step > 0 {
                preconditioned -= &previous * (off_diagonal / previous_off)
            }
            let diagonal_entry = basis.full_contraction(&preconditioned);
            preconditioned -= &current * (diagonal_entry / off_diagonal);
            previous = std::mem::replace(&mut current, preconditioned);
            preconditioned = divide(&current);
            previous_off = off_diagonal;
            off_diagonal = current.full_contraction(&preconditioned).max(0.0).sqrt();
            //
            // The rotation of the iteration before reaches two entries ahead, so
            // what it left behind is applied before a rotation of this one is
            // found to annihilate the entry below the diagonal.
            //
            let reached = carried;
            let shifted = cosine * trailing + sine * diagonal_entry;
            let remaining = sine * trailing - cosine * diagonal_entry;
            carried = sine * off_diagonal;
            trailing = -cosine * off_diagonal;
            let rotated = remaining
                .hypot(off_diagonal)
                .max(Scalar::EPSILON * scale.max(1.0));
            cosine = remaining / rotated;
            sine = off_diagonal / rotated;
            older = std::mem::replace(&mut old, direction);
            direction = (basis - &older * reached - &old * shifted) * rotated.recip();
            solution += &direction * (cosine * length);
            length *= sine;
            if length.abs() <= self.rel_tol * scale {
                return Ok(solution);
            }
        }
        Err(KrylovError::MaximumStepsReached(
            self.max_steps,
            length.abs() / scale,
        ))
    }
}

/// Possible errors encountered during an iterative linear solve.
pub enum KrylovError {
    MaximumStepsReached(usize, Scalar),
    NotPositiveDefinite(Scalar),
}

impl StyledError for KrylovError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::MaximumStepsReached(steps, relative) => format!(
                "{h}Maximum number of iterations ({steps}) reached.{c}\n\
                Residual relative to the one started from: {relative:?}."
            ),
            Self::NotPositiveDefinite(curvature) => format!(
                "{h}The tangent is not positive definite.{c}\n\
                Curvature along a direction: {curvature:?}."
            ),
        }
    }
}

styled_error!(KrylovError);

impl From<KrylovError> for OptimizationError {
    fn from(error: KrylovError) -> Self {
        Self::Upstream(error.to_string(), "Krylov".to_string())
    }
}

impl From<KrylovError> for AssertionError {
    fn from(error: KrylovError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
