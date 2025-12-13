use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{
        NodalCoordinatesBlock, NodalForcesSolid, NodalStiffnessesSolid,
        block::{
            ElementBlock, FiniteElementBlockError, FirstOrderMinimize, SecondOrderMinimize, band,
            element::HyperelasticFiniteElement, solid::elastic::ElasticFiniteElementBlock,
        },
    },
    math::{
        Scalar, Tensor,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, OptimizationError, SecondOrderOptimization,
        },
    },
};

pub trait HyperelasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
    Self: ElasticFiniteElementBlock<C, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, FiniteElementBlockError>;
}

impl<C, F, const G: usize, const N: usize> HyperelasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
    Self: ElasticFiniteElementBlock<C, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, F, const G: usize, const N: usize> FirstOrderMinimize<C, F, G, N, NodalCoordinatesBlock>
    for ElementBlock<C, F, N>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, NodalForcesSolid>,
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
        solver.minimize(
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

impl<C, F, const G: usize, const N: usize>
    SecondOrderMinimize<C, F, G, N, NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinatesBlock>
    for ElementBlock<C, F, N>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinatesBlock,
        >,
    ) -> Result<NodalCoordinatesBlock, OptimizationError> {
        let banded = band(
            self.connectivity(),
            &equality_constraint,
            self.coordinates().len(),
            3,
        );
        solver.minimize(
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinatesBlock| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinatesBlock| {
                Ok(self.nodal_stiffnesses(nodal_coordinates)?)
            },
            self.coordinates().clone().into(),
            equality_constraint,
            Some(banded),
        )
    }
}
