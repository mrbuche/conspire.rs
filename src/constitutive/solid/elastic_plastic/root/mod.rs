#[cfg(test)]
mod test;

use super::{AppliedLoad, ElasticPlastic, load::Load};
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariablesHistory},
    math::{
        Matrix, Quantity, TensorArray, Vector,
        integrate::StepController,
        optimize::{RootFindingBlock, SolveStrategy},
    },
    mechanics::{
        DeformationGradient, DeformationGradients, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Scalar, Times,
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

const DT_CUT: Scalar = 0.2;

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
        if time.len() < 2 || time[0] >= time[time.len() - 1] {
            return Err(ConstitutiveError::custom(
                "The times must be at least two and increasing.",
                self,
            ));
        }
        let (t_0, t_f) = (time[0], time[time.len() - 1]);
        let direction = StepController::new(0.9, 2.0, DT_CUT, 5.0);
        let overshoot = StepController::new(0.9, 1.0, DT_CUT, 5.0);
        let dt_min = (t_f - t_0) * 1e-10;
        let mut t = t_0;
        let mut dt = t_f - t_0;
        let mut breakpoint = 1;
        let mut state = self.initial_state();
        let mut deformation_gradient = DeformationGradient::identity();
        let mut times = vec![t_0];
        let mut deformation_gradients = vec![deformation_gradient.clone()];
        let mut states = vec![state.clone()];
        while breakpoint < time.len() {
            let remaining = time[breakpoint] - t;
            let landing = dt >= remaining;
            let step = if landing { remaining } else { dt };
            let t_next = if landing { time[breakpoint] } else { t + step };
            let advanced = load.advance(
                self,
                &solver,
                &strategy,
                t_next,
                &deformation_gradient,
                &state,
            );
            let (deformation_gradient_next, state_next) = match advanced {
                Ok(next) => next,
                Err(error) => {
                    if step <= dt_min {
                        return Err(error);
                    }
                    dt = step * DT_CUT;
                    continue;
                }
            };
            let error = self.step_error(
                &deformation_gradient,
                &deformation_gradient_next,
                &state,
                &state_next,
            )?;
            let scale = direction
                .scale(error.direction, abs_tol)
                .min(overshoot.scale(error.overshoot, abs_tol));
            if error.direction <= abs_tol && error.overshoot <= abs_tol {
                t = t_next;
                deformation_gradient = deformation_gradient_next;
                state = state_next;
                if landing {
                    breakpoint += 1;
                }
                if landing || time.len() == 2 {
                    times.push(t);
                    deformation_gradients.push(deformation_gradient.clone());
                    states.push(state.clone());
                }
                dt = if landing {
                    dt.max(step * scale)
                } else {
                    step * scale
                };
            } else if step <= dt_min {
                return Err(ConstitutiveError::custom(
                    "The adaptive step fell below the floor.",
                    self,
                ));
            } else {
                dt = step * scale;
            }
        }
        Ok((
            times.into_iter().collect(),
            deformation_gradients.into(),
            states.into(),
        ))
    }
}
