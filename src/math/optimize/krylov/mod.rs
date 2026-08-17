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

/// An iterative linear solve, asking the tangent only what it does to a vector.
///
/// Nothing is factorized and nothing is assembled, so the tangent is free never
/// to exist as a matrix at all. What that costs is exactness: the solve stops
/// once the residual it leaves behind is small enough against the one it started
/// from, rather than at an answer.
#[derive(Clone, Copy, Debug)]
pub struct Krylov {
    /// Maximum number of iterations.
    pub max_steps: usize,
    /// Preconditioner.
    pub preconditioner: Preconditioner,
    /// Residual tolerance, relative to the residual started from.
    pub rel_tol: Scalar,
}

impl Default for Krylov {
    fn default() -> Self {
        Self {
            max_steps: 1_000,
            preconditioner: Preconditioner::default(),
            rel_tol: 1e-10,
        }
    }
}

impl Krylov {
    /// The diagonal to divide through by, at the given positions of a tangent.
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
                    .map(|i| match tangent.entry(i, i) {
                        0.0 => 1.0,
                        entry => entry,
                    })
                    .collect(),
            ),
            Preconditioner::None => None,
        }
    }
    /// Solves a system by the method of conjugate gradients.
    pub fn conjugate_gradients<H>(
        &self,
        tangent: &H,
        right_hand_side: &Vector,
    ) -> Result<Vector, KrylovError>
    where
        H: Hessian,
    {
        self.descend(
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
    pub fn conjugate_gradients_retained<H>(
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
        self.descend(
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
        let divide = |residual: &Vector| match &diagonal {
            Some(diagonal) => residual
                .iter()
                .zip(diagonal.iter())
                .map(|(entry, scale)| entry / scale)
                .collect(),
            None => residual.clone(),
        };
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
