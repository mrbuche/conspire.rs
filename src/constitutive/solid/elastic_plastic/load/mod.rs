#[cfg(test)]
mod test;

use super::{AppliedLoad, ElasticPlastic, Prescribed, bcs, coupled};
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariables},
    math::{
        Matrix, Quantity, Vector,
        optimize::{RootFindingBlock, SolveStrategy},
        sparse::CscMatrix,
    },
    mechanics::{
        DeformationGradient, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
    },
    units::Time,
};

pub(super) struct Load {
    matrix: CscMatrix,
    vector: Vector,
    prescribed: Prescribed,
    constraint: (CscMatrix, Vector),
}

impl Load {
    pub(super) fn new(applied_load: AppliedLoad<'_>) -> (Self, &[Quantity<Time>]) {
        let (matrix, prescribed, time) = bcs(applied_load);
        let mut pattern = Vec::new();
        for row in 0..matrix.len() {
            for column in 0..9 {
                if matrix[row][column] != 0.0 {
                    pattern.push((row, column));
                }
            }
        }
        let mut global = CscMatrix::from_pattern(matrix.len(), 9, pattern);
        global.fill(|row, column| matrix[row][column]);
        (
            Self {
                matrix: global,
                vector: Vector::zero(matrix.len()),
                prescribed,
                constraint: (
                    CscMatrix::from_pattern(0, coupled::SIZE, Vec::new()),
                    Vector::zero(0),
                ),
            },
            time,
        )
    }
    pub(super) fn advance<C: ElasticPlastic>(
        &mut self,
        model: &C,
        solver: &impl RootFindingBlock<
            DeformationGradient,
            Vector,
            FirstPiolaKirchhoffStress,
            Vector,
            FirstPiolaKirchhoffTangentStiffness,
            Matrix,
            Matrix,
            Matrix,
        >,
        strategy: &SolveStrategy,
        time: Quantity<Time>,
        deformation_gradient: &DeformationGradient,
        previous_state: &PlasticStateVariables,
    ) -> Result<(DeformationGradient, PlasticStateVariables), ConstitutiveError> {
        self.prescribed
            .iter()
            .for_each(|(index, function)| self.vector[*index] = function(time));
        let (deformation_gradient_new, local_new) = solver
            .root_block(
                |global: &DeformationGradient, local: &Vector| {
                    coupled::monolithic_plastic(model, previous_state, local)
                        .and_then(|plastic| model.first_piola_kirchhoff_stress(global, &plastic))
                        .map_err(|error| error.to_string())
                },
                |global: &DeformationGradient, local: &Vector| {
                    coupled::monolithic_residual_local(model, global, previous_state, local)
                        .map_err(|error| error.to_string())
                },
                |global: &DeformationGradient, local: &Vector| {
                    coupled::monolithic_tangents(model, global, previous_state, local)
                        .map_err(|error| error.to_string())
                },
                (deformation_gradient.clone(), Vector::zero(coupled::SIZE)),
                (self.matrix.clone(), self.vector.clone()),
                self.constraint.clone(),
                None,
                strategy.clone(),
            )
            .map_err(|error| ConstitutiveError::upstream(error, model))?;
        Ok((
            deformation_gradient_new,
            coupled::monolithic_state(model, previous_state, &local_new)?,
        ))
    }
}
