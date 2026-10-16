use crate::{
    constitutive::diffusion::{Diffusion, Diffusive},
    domain::diffusion::{DiffusionElements, NodalFields, NodalForces, NodalStiffnesses},
    fem::{
        ElementModelError,
        block::{
            Block,
            element::{
                FiniteElementError,
                diffusion::{
                    DiffusionFiniteElement, DiffusiveFiniteElement, ElementGradients,
                    ElementNodalFields,
                },
            },
        },
    },
    math::Quantity,
};

pub trait DiffusiveElements<U, C, F, const G: usize, const M: usize, const N: usize, const P: usize>
where
    U: Diffusive,
    F: DiffusiveFiniteElement<U, G, M, N, P>,
{
    fn nodal_fields_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_fields: &NodalFields<U>,
    ) -> ElementNodalFields<U, N>;
    fn gradients(&self, nodal_fields: &NodalFields<U>) -> Vec<ElementGradients<U, G>>;
}

impl<U, C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    DiffusiveElements<U, C, F, G, M, N, P> for Block<C, F, G, M, N, P, R, H>
where
    U: Diffusive,
    F: DiffusiveFiniteElement<U, G, M, N, P>,
{
    fn nodal_fields_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_fields: &NodalFields<U>,
    ) -> ElementNodalFields<U, N> {
        element_connectivity
            .iter()
            .map(|&node| nodal_fields[node])
            .collect()
    }
    fn gradients(&self, nodal_fields: &NodalFields<U>) -> Vec<ElementGradients<U, G>> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, element_connectivity)| {
                element.gradients(&self.nodal_fields_element(element_connectivity, nodal_fields))
            })
            .collect()
    }
}

impl<U, C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    DiffusionElements<U> for Block<C, F, G, M, N, P, R, H>
where
    U: Diffusive,
    C: Diffusion<Field = U>,
    F: DiffusionFiniteElement<C, G, M, N, P>,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<U>,
    ) -> Result<Quantity<U::Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, element_connectivity)| {
                element.potential(
                    self.constitutive_model(),
                    &self.nodal_fields_element(element_connectivity, nodal_fields),
                )
            })
            .sum::<Result<Quantity<U::Energy>, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_forces: &mut NodalForces<U>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &self.nodal_fields_element(element_connectivity, nodal_fields),
                    )?
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_stiffnesses: &mut NodalStiffnesses<U>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &self.nodal_fields_element(element_connectivity, nodal_fields),
                    )?
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(object, &node_a)| {
                        object.into_iter().zip(element_connectivity).for_each(
                            |(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            },
                        )
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
