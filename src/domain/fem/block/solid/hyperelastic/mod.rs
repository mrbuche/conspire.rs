use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{
        NodalCoordinates,
        block::{
            ElementBlock, FiniteElementBlockError, FirstOrderMinimize, SecondOrderMinimize, band,
            element::solid::hyperelastic::HyperelasticFiniteElement,
            solid::{
                NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElementBlock,
                elastic::ElasticFiniteElementBlock,
            },
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
        nodal_coordinates: &NodalCoordinates,
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
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &self.element_nodal_coordinates(element_connectivity, nodal_coordinates),
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

impl<C, F, const G: usize, const N: usize> FirstOrderMinimize<C, F, G, N, NodalCoordinates>
    for ElementBlock<C, F, N>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, N>,
{
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, NodalForcesSolid>,
    ) -> Result<NodalCoordinates, OptimizationError> {
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_forces(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
        )
    }
}

impl<C, F, const G: usize, const N: usize>
    SecondOrderMinimize<C, F, G, N, NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>
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
            NodalCoordinates,
        >,
    ) -> Result<NodalCoordinates, OptimizationError> {
        let banded = band(
            self.connectivity(),
            &equality_constraint,
            self.coordinates().len(),
            3,
        );
        solver.minimize(
            |nodal_coordinates: &NodalCoordinates| {
                Ok(self.helmholtz_free_energy(nodal_coordinates)?)
            },
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_forces(nodal_coordinates)?),
            |nodal_coordinates: &NodalCoordinates| Ok(self.nodal_stiffnesses(nodal_coordinates)?),
            self.coordinates().clone().into(),
            equality_constraint,
            Some(banded),
        )
    }
}
