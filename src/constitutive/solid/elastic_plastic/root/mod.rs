#[cfg(test)]
mod test;

use super::{AppliedLoad, ElasticPlastic, load::Load};
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariablesHistory},
    math::{
        Matrix, Quantity, TensorArray, Vector,
        integrate::adapt,
        optimize::{RootFindingBlock, SolveStrategy},
    },
    mechanics::{
        DeformationGradient, DeformationGradients, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Times,
    },
};

/// Root-finding methods for elastic-plastic solid constitutive models.
pub trait Root {
    /// Solve for the unknown components of the deformation gradients under an applied load.
    ///
    /// With [`SolveStrategy::Condensed`], the coupled local unknowns
    /// $`(\mathbf{E},\Delta\gamma)`$ of the return map are converged at each outer
    /// iterate and eliminated from the tangent by a Schur complement.
    ///
    /// [`SolveStrategy::Monolithic`] steps the deformation gradient and the coupled local
    /// unknowns $`(\mathbf{E},\Delta\gamma)`$ of the return map together through
    /// [`RootFindingBlock::root_block`], rather than converging the local
    /// block before every outer step. The local residual is the same coupled system the
    /// return map solves, with the yield inequality imposed by a Fischer-Burmeister
    /// complementarity residual so elastic steps recover $`\Delta\gamma = 0`$ on their
    /// own. It converges to the same step as [`SolveStrategy::Condensed`], also under
    /// non-proportional loading.
    ///
    /// The model is rate independent, so the times of the load are values of the load
    /// parameter, and the step is chosen by [`StepError`](crate::math::integrate::StepError). Two times give only the span, and
    /// the accepted steps are reported. More than two are breakpoints: the steps are cut to
    /// end on each, and only they are reported, since the response has kinks at yielding
    /// and at unloading that an interpolant between steps would smooth over.
    ///
    /// A step is accepted when both the estimated error of the step in the plastic
    /// deformation, from the change of the flow direction over it, and the plastic increment
    /// of a step that begins inside the yield surface are within `abs_tol`, in units of the
    /// yield strain, so the tolerance is a fraction of the reversible range of the response.
    /// The first falls as the square of the step and the second as the step. A step the
    /// solver fails on is cut back like a rejected one.
    ///
    /// The tolerance bounds the estimated error of each step, as those of the integrators
    /// do, so the errors add up over the steps: the error of the result is larger than
    /// `abs_tol` and falls more slowly than it.
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl RootFindingBlock<
            DeformationGradient,
            Vector,
            FirstPiolaKirchhoffStress,
            Vector,
            FirstPiolaKirchhoffTangentStiffness,
            Matrix,
            Matrix,
            Matrix,
        >,
        strategy: SolveStrategy,
        abs_tol: Quantity,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>;
}

impl<C> Root for C
where
    C: ElasticPlastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        solver: impl RootFindingBlock<
            DeformationGradient,
            Vector,
            FirstPiolaKirchhoffStress,
            Vector,
            FirstPiolaKirchhoffTangentStiffness,
            Matrix,
            Matrix,
            Matrix,
        >,
        strategy: SolveStrategy,
        abs_tol: Quantity,
    ) -> Result<(Times, DeformationGradients, PlasticStateVariablesHistory), ConstitutiveError>
    {
        let (mut load, time) = Load::new(applied_load);
        let reported = adapt(
            time,
            abs_tol,
            (DeformationGradient::identity(), self.initial_state()),
            |time, deformation_gradient, state| {
                load.advance(self, &solver, &strategy, time, deformation_gradient, state)
            },
            |deformation_gradient, deformation_gradient_next, state, state_next| {
                Ok(self
                    .step_error(
                        deformation_gradient,
                        deformation_gradient_next,
                        state,
                        state_next,
                    )?
                    .relative_to(self.yield_strain(state)?))
            },
            |error| ConstitutiveError::custom(error, self),
        )?;
        let mut times = Vec::new();
        let mut deformation_gradients = Vec::new();
        let mut states = Vec::new();
        for (time, deformation_gradient, state) in reported {
            times.push(time);
            deformation_gradients.push(deformation_gradient);
            states.push(state);
        }
        Ok((
            times.into_iter().collect(),
            deformation_gradients.into(),
            states.into(),
        ))
    }
}
