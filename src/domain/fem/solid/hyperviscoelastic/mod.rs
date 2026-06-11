use crate::{
    fem::{
        Blocks, FiniteElementModelError, Model, NodalCoordinates,
        solid::elastic_hyperviscous::ElasticHyperviscousFiniteElements,
    },
    math::Scalar,
};

pub trait HyperviscoelasticFiniteElements<const D: usize>
where
    Self: ElasticHyperviscousFiniteElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Scalar, FiniteElementModelError>;
}

impl<B, const D: usize> HyperviscoelasticFiniteElements<D> for Model<B, D>
where
    B: HyperviscoelasticFiniteElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks.helmholtz_free_energy(nodal_coordinates)
    }
}

impl<B1, B2, const D: usize> HyperviscoelasticFiniteElements<D> for Blocks<B1, B2>
where
    B1: HyperviscoelasticFiniteElements<D>,
    B2: HyperviscoelasticFiniteElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self.0.helmholtz_free_energy(nodal_coordinates)?
            + self.1.helmholtz_free_energy(nodal_coordinates)?)
    }
}
