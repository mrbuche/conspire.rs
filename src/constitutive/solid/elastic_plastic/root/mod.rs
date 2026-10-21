#[cfg(test)]
mod test;

use super::{AppliedLoad, ElasticPlastic, Outcome, Stepper, load::Load};
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariablesHistory},
    math::{
        Matrix, Quantity, TensorArray, Vector,
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
    /// parameter, and the step is chosen by [`StepError`](super::StepError). Two times give only the span, and
    /// the accepted steps are reported. More than two are breakpoints: the steps are cut to
    /// end on each, and only they are reported, since the response has kinks at yielding
    /// and at unloading that an interpolant between steps would smooth over.
    ///
    /// A step is accepted when both the change of the flow direction over it and the
    /// plastic increment of a step that begins inside the yield surface are within
    /// `abs_tol`, strains. The first falls as the square of the step and the second as the
    /// step. A step the solver fails on is cut back like a rejected one.
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
        let mut stepper =
            Stepper::new(time).map_err(|error| ConstitutiveError::custom(error, self))?;
        let mut state = self.initial_state();
        let mut deformation_gradient = DeformationGradient::identity();
        let mut times = vec![time[0]];
        let mut deformation_gradients = vec![deformation_gradient.clone()];
        let mut states = vec![state.clone()];
        while !stepper.is_done() {
            let step = stepper.propose();
            let (deformation_gradient_next, state_next) = match load.advance(
                self,
                &solver,
                &strategy,
                step.time,
                &deformation_gradient,
                &state,
            ) {
                Ok(next) => next,
                Err(_) if stepper.fail(&step) => continue,
                Err(error) => return Err(error),
            };
            let error = self.step_error(
                &deformation_gradient,
                &deformation_gradient_next,
                &state,
                &state_next,
            )?;
            match stepper
                .judge(&step, error, abs_tol)
                .map_err(|error| ConstitutiveError::custom(error, self))?
            {
                Outcome::Accepted { report } => {
                    deformation_gradient = deformation_gradient_next;
                    state = state_next;
                    if report {
                        times.push(step.time);
                        deformation_gradients.push(deformation_gradient.clone());
                        states.push(state.clone());
                    }
                }
                Outcome::Rejected => {}
            }
        }
        Ok((
            times.into_iter().collect(),
            deformation_gradients.into(),
            states.into(),
        ))
    }
}
