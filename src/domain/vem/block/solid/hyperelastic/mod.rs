use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{ElementModelError, solid::hyperelastic::HyperelasticElements},
    math::{HessianAccumulate, Scalar},
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{VirtualElementError, solid::hyperelastic::HyperelasticVirtualElement},
            solid::NodalStiffnessesSolidSymmetric,
        },
    },
};

impl<C, F> HyperelasticElements<3> for Block<C, F>
where
    C: Hyperelastic,
    F: HyperelasticVirtualElement<C>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    Self::element_coordinates(nodal_coordinates, nodes),
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses_symmetric_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_stiffnesses: &mut NodalStiffnessesSolidSymmetric,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                if node_a <= node_b {
                                    nodal_stiffnesses.accumulate(node_a, node_b, nodal_stiffness)
                                }
                            })
                    });
                Ok::<(), VirtualElementError>(())
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
