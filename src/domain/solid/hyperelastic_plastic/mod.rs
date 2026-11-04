use crate::{
    domain::{
        ElementModelError, Model, NodalCoordinates, solid::elastic_plastic::ElasticPlasticElements,
    },
    math::Quantity,
    units::Energy,
};

pub trait HyperelasticPlasticElements<S, const D: usize>
where
    Self: ElasticPlasticElements<S, D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Quantity<Energy>, ElementModelError>;
}

impl<B, S, const D: usize> HyperelasticPlasticElements<S, D> for Model<B, D>
where
    B: HyperelasticPlasticElements<S, D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.blocks
            .helmholtz_free_energy(nodal_coordinates, state_variables)
    }
}
