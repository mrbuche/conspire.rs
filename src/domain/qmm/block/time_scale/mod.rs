use super::Block;
use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{
        ElementModelError, NodalCoordinates,
        mass::{LumpedMassElements, NodalLumpedMasses},
        solid::elastic::ElasticElements,
        time_scale::{
            TimeScaleElements, largest_eigenvalue_by_product, time_scale_from_eigenvalue,
        },
    },
    math::{Quantity, Scalar, Tensor},
    units::{Density, Time},
};

impl<C> TimeScaleElements<3> for Block<C, Quantity<Density>>
where
    C: Elastic,
{
    fn fastest_time_scale(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        let stiffnesses = self.nodal_stiffnesses(nodal_coordinates)?;
        let mut lumped_masses = NodalLumpedMasses::zero(nodal_coordinates.len());
        self.nodal_lumped_masses_into(&mut lumped_masses);
        let masses: Vec<Scalar> = lumped_masses
            .iter()
            .flat_map(|mass| [mass.value(); 3])
            .collect();
        Ok(time_scale_from_eigenvalue(largest_eigenvalue_by_product(
            masses.len(),
            |vector| {
                let mut product = vec![0.0; vector.len()];
                stiffnesses.iter().enumerate().for_each(|(a, row)| {
                    row.entries().for_each(|(b, block)| {
                        (0..3).for_each(|i| {
                            (0..3).for_each(|j| {
                                product[3 * a + i] += block[i][j].value() * vector[3 * b + j]
                            })
                        })
                    })
                });
                product
            },
            &masses,
        )))
    }
}
