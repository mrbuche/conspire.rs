#[cfg(test)]
mod test;

use super::*;

const G: usize = 1;
const M: usize = 3;
const N: usize = 4;
const O: usize = 4;

const INTEGRATION_WEIGHT: Scalar = ONE_SIXTH;

const STANDARD_GRADIENT_OPERATOR: StandardGradientOperator<M, O> = TensorRank1List([
    TensorRank1([-1.0, -1.0, -1.0]),
    TensorRank1([1.0, 0.0, 0.0]),
    TensorRank1([0.0, 1.0, 0.0]),
    TensorRank1([0.0, 0.0, 1.0]),
]);

pub struct Tetrahedron<C> {
    constitutive_model: C,
    gradient_vectors: GradientVectors<N>,
    integration_weight: Scalar,
}

impl<'a, C> FiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Constitutive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self {
        Self {
            constitutive_model: <C>::new(constitutive_model_parameters),
            gradient_vectors: Self::calculate_gradient_vectors(&reference_nodal_coordinates),
            integration_weight: Self::calculate_reference_jacobian(&reference_nodal_coordinates)
                * INTEGRATION_WEIGHT,
        }
    }
}

impl<'a, C> LinearElement<'a, C, G, M, N, O> for Tetrahedron<C>
where
    C: Constitutive<'a>,
{
    fn calculate_gradient_vectors(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> GradientVectors<N> {
        (reference_nodal_coordinates * &STANDARD_GRADIENT_OPERATOR).inverse_transpose()
            * STANDARD_GRADIENT_OPERATOR
    }
    fn calculate_reference_jacobian(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> Scalar {
        (reference_nodal_coordinates * STANDARD_GRADIENT_OPERATOR).determinant()
    }
    fn calculate_standard_gradient_operator() -> StandardGradientOperator<M, O> {
        STANDARD_GRADIENT_OPERATOR
    }
    fn get_constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn get_gradient_vectors(&self) -> &GradientVectors<N> {
        &self.gradient_vectors
    }
    fn get_integration_weight(&self) -> &Scalar {
        &self.integration_weight
    }
}

impl<'a, C> ElasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Elastic<'a>,
{
    fn calculate_deformations(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradients<G> {
        TensorRank2List([self.calculate_deformation_gradient(nodal_coordinates)])
    }
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        self.calculate_nodal_forces_linear_element(nodal_coordinates)
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        self.calculate_nodal_stiffnesses_linear_element(nodal_coordinates)
    }
}
impl<'a, C> ElasticLinearElement<'a, C, G, M, N, O> for Tetrahedron<C> where C: Elastic<'a> {}
impl<'a, C> HyperelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Hyperelastic<'a>,
{
    fn calculate_helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_helmholtz_free_energy_linear_element(nodal_coordinates)
    }
}
impl<'a, C> HyperelasticLinearElement<'a, C, G, M, N, O> for Tetrahedron<C> where C: Hyperelastic<'a>
{}
impl<'a, C> ViscoelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Viscoelastic<'a>,
{
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        self.calculate_nodal_forces_linear_element(nodal_coordinates, nodal_velocities)
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        self.calculate_nodal_stiffnesses_linear_element(nodal_coordinates, nodal_velocities)
    }
}
impl<'a, C> ViscoelasticLinearElement<'a, C, G, M, N, O> for Tetrahedron<C> where C: Viscoelastic<'a>
{}
impl<'a, C> ElasticHyperviscousFiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: ElasticHyperviscous<'a>,
{
    fn calculate_viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_viscous_dissipation_linear_element(nodal_coordinates, nodal_velocities)
    }
    fn calculate_dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_dissipation_potential_linear_element(nodal_coordinates, nodal_velocities)
    }
}
impl<'a, C> ElasticHyperviscousLinearElement<'a, C, G, M, N, O> for Tetrahedron<C> where
    C: ElasticHyperviscous<'a>
{
}
impl<'a, C> HyperviscoelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Hyperviscoelastic<'a>,
{
    fn calculate_helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_helmholtz_free_energy_linear_element(nodal_coordinates)
    }
}
impl<'a, C> HyperviscoelasticLinearElement<'a, C, G, M, N, O> for Tetrahedron<C> where
    C: Hyperviscoelastic<'a>
{
}
