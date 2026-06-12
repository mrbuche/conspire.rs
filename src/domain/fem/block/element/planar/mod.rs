use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{elastic::Elastic, hyperelastic::Hyperelastic},
    },
    fem::block::element::{
        Element, FiniteElement, FiniteElementError, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients, basic_from,
        surface::SurfaceElement,
    },
    math::{Scalar, ScalarList, Tensor, TensorArray, TensorRank1List, TensorRank2List2D},
    mechanics::{
        DeformationGradient, DeformationGradientList, FirstPiolaKirchhoffStressList,
        FirstPiolaKirchhoffTangentStiffnessList,
    },
};

const M: usize = 2;

pub type Quadrilateral = Element<2, 4, 4, 1>;
pub type Triangle = Element<2, 1, 3, 1>;

pub type PlanarElementNodalCoordinates<const N: usize> = TensorRank1List<M, 1, N>;
pub type PlanarElementNodalReferenceCoordinates<const N: usize> = TensorRank1List<M, 0, N>;
pub type PlanarElementNodalForcesSolid<const N: usize> = TensorRank1List<M, 1, N>;
pub type PlanarElementNodalStiffnessesSolid<const N: usize> = TensorRank2List2D<M, 1, 1, N, N>;

impl<const G: usize, const N: usize, const O: usize, const P: usize> FiniteElement<G, M, N, P>
    for Element<2, G, N, O>
where
    SurfaceElement<G, N, O>: FiniteElement<G, M, N, P>,
{
    fn integration_points() -> ParametricCoordinates<G, M> {
        SurfaceElement::<G, N, O>::integration_points()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        SurfaceElement::<G, N, O>::parametric_reference()
    }
    fn parametric_weights() -> ScalarList<G> {
        SurfaceElement::<G, N, O>::parametric_weights()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<P> {
        SurfaceElement::<G, N, O>::shape_functions(parametric_coordinate)
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, P> {
        SurfaceElement::<G, N, O>::shape_functions_gradients(parametric_coordinate)
    }
}

impl<const G: usize, const N: usize, const O: usize> From<PlanarElementNodalReferenceCoordinates<N>>
    for Element<2, G, N, O>
where
    Self: FiniteElement<G, M, N, N>,
{
    fn from(reference_nodal_coordinates: PlanarElementNodalReferenceCoordinates<N>) -> Self {
        basic_from(reference_nodal_coordinates)
    }
}

pub trait PlanarSolidFiniteElement<const G: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, M, N, P>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &PlanarElementNodalCoordinates<N>,
    ) -> DeformationGradientList<G>;
}

impl<const G: usize, const N: usize, const O: usize, const P: usize>
    PlanarSolidFiniteElement<G, N, P> for Element<2, G, N, O>
where
    Self: FiniteElement<G, M, N, P>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &PlanarElementNodalCoordinates<N>,
    ) -> DeformationGradientList<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                let mut deformation_gradient = DeformationGradient::zero();
                deformation_gradient[2][2] = 1.0;
                nodal_coordinates.iter().zip(gradient_vectors).for_each(
                    |(nodal_coordinate, gradient_vector)| {
                        (0..M).for_each(|i| {
                            (0..M).for_each(|j| {
                                deformation_gradient[i][j] +=
                                    nodal_coordinate[i] * gradient_vector[j]
                            })
                        })
                    },
                );
                deformation_gradient
            })
            .collect()
    }
}

pub trait PlanarElasticFiniteElement<C, const G: usize, const N: usize, const P: usize>
where
    C: Elastic,
    Self: PlanarSolidFiniteElement<G, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &PlanarElementNodalCoordinates<N>,
    ) -> Result<PlanarElementNodalForcesSolid<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &PlanarElementNodalCoordinates<N>,
    ) -> Result<PlanarElementNodalStiffnessesSolid<N>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    PlanarElasticFiniteElement<C, G, N, P> for Element<2, G, N, O>
where
    C: Elastic,
    Self: PlanarSolidFiniteElement<G, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &PlanarElementNodalCoordinates<N>,
    ) -> Result<PlanarElementNodalForcesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffStressList<G>, _>>()
        {
            Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
                .iter()
                .zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights()),
                )
                .map(
                    |(first_piola_kirchhoff_stress, (gradient_vectors, integration_weight))| {
                        gradient_vectors
                            .iter()
                            .map(|gradient_vector| {
                                (0..M)
                                    .map(|i| {
                                        (0..M)
                                            .map(|j| {
                                                first_piola_kirchhoff_stress[i][j]
                                                    * gradient_vector[j]
                                            })
                                            .sum::<Scalar>()
                                            * integration_weight
                                    })
                                    .collect()
                            })
                            .collect()
                    },
                )
                .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &PlanarElementNodalCoordinates<N>,
    ) -> Result<PlanarElementNodalStiffnessesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnessList<G>, _>>()
        {
            Ok(first_piola_kirchhoff_tangent_stiffnesses) => {
                Ok(first_piola_kirchhoff_tangent_stiffnesses
                    .iter()
                    .zip(
                        self.gradient_vectors()
                            .iter()
                            .zip(self.integration_weights()),
                    )
                    .map(
                        |(
                            first_piola_kirchhoff_tangent_stiffness,
                            (gradient_vectors, integration_weight),
                        )| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_a| {
                                    gradient_vectors
                                        .iter()
                                        .map(|gradient_vector_b| {
                                            (0..M)
                                                .map(|i| {
                                                    (0..M)
                                                        .map(|k| {
                                                            (0..M)
                                                                .map(|j| {
                                                                    (0..M)
                                                                        .map(|l| {
                                                                            first_piola_kirchhoff_tangent_stiffness
                                                                                [i][j][k][l]
                                                                                * gradient_vector_a[j]
                                                                                * gradient_vector_b[l]
                                                                        })
                                                                        .sum::<Scalar>()
                                                                })
                                                                .sum::<Scalar>()
                                                                * integration_weight
                                                        })
                                                        .collect()
                                                })
                                                .collect()
                                        })
                                        .collect()
                                })
                                .collect()
                        },
                    )
                    .sum())
            }
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

pub trait PlanarHyperelasticFiniteElement<C, const G: usize, const N: usize, const P: usize>
where
    C: Hyperelastic,
    Self: PlanarElasticFiniteElement<C, G, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &PlanarElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    PlanarHyperelasticFiniteElement<C, G, N, P> for Element<2, G, N, O>
where
    C: Hyperelastic,
    Self: PlanarElasticFiniteElement<C, G, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &PlanarElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.integration_weights())
            .map(|(deformation_gradient, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                        * integration_weight,
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
