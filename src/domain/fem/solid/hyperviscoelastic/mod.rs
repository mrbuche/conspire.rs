use crate::{
    fem::{
        Blocks, FiniteElementModelError, Model, NodalCoordinates,
        solid::elastic_hyperviscous::ElasticHyperviscousFiniteElements,
    },
    math::Scalar,
};

pub trait HyperviscoelasticFiniteElements
where
    Self: ElasticHyperviscousFiniteElements,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementModelError>;
}

impl<B> HyperviscoelasticFiniteElements for Model<B>
where
    B: HyperviscoelasticFiniteElements,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks.helmholtz_free_energy(nodal_coordinates)
    }
}

impl<B1, B2> HyperviscoelasticFiniteElements for Blocks<B1, B2>
where
    B1: HyperviscoelasticFiniteElements,
    B2: HyperviscoelasticFiniteElements,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self.0.helmholtz_free_energy(nodal_coordinates)?
            + self.1.helmholtz_free_energy(nodal_coordinates)?)
    }
}
