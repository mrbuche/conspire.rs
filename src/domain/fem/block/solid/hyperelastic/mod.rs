use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{
        NodalCoordinates,
        block::{
            Block, FiniteElementBlockError, FirstOrderMinimize, SecondOrderMinimize, band,
            element::solid::hyperelastic::HyperelasticFiniteElement,
            solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticFiniteElementBlock},
        },
    },
    math::{
        Scalar, Tensor,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, OptimizationError, SecondOrderOptimization,
        },
    },
};

pub trait HyperelasticFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
    Self: ElasticFiniteElementBlock<C, F, G, M, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementBlockError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    HyperelasticFiniteElementBlock<C, F, G, M, N, P> for Block<C, F, G, M, N, P>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
    Self: ElasticFiniteElementBlock<C, F, G, M, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
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

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    FirstOrderMinimize<C, F, G, M, N, NodalCoordinates> for Block<C, F, G, M, N, P>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
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

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    SecondOrderMinimize<C, F, G, M, N, NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>
    for Block<C, F, G, M, N, P>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
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
