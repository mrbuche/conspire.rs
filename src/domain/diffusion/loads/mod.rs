use crate::{
    constitutive::diffusion::Diffusive,
    domain::{
        boundary::{Conservative, ConservativeAt, Field, Load, LoadAt, Loads},
        diffusion::{NodalFields, NodalForces},
    },
    math::{Quantity, Tensor},
    units::Time,
};
use std::ops::Mul;

/// Sources that do not change with the field, one on each node, such as heat or charge.
#[derive(Clone, Debug)]
pub struct NodalSource<L>
where
    L: Diffusive,
{
    sources: NodalForces<L>,
}

impl<U, T> Loads<Field<U>, T> {
    /// Adds a source on each node, such as the heating of a body.
    pub fn source<L>(self, sources: NodalForces<L>) -> Loads<Field<U>, (T, NodalSource<L>)>
    where
        L: Diffusive<Field = U>,
    {
        self.with(NodalSource { sources })
    }
}

impl<L> NodalSource<L>
where
    L: Diffusive,
{
    fn check(&self, nodes: usize) -> Result<(), String> {
        if nodes == self.sources.len() {
            Ok(())
        } else {
            Err("The source is on a mesh with another number of nodes.".to_string())
        }
    }
}

impl<L> Load<NodalFields<L>, NodalForces<L>> for NodalSource<L>
where
    L: Diffusive,
{
    fn subtract_from(
        &self,
        nodal_fields: &NodalFields<L>,
        residual: &mut NodalForces<L>,
    ) -> Result<(), String> {
        self.check(nodal_fields.len())?;
        residual
            .iter_mut()
            .zip(&self.sources)
            .for_each(|(residual, source)| *residual -= source);
        Ok(())
    }
}

impl<L> LoadAt<NodalFields<L>, NodalForces<L>> for NodalSource<L>
where
    L: Diffusive,
{
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
        residual: &mut NodalForces<L>,
    ) -> Result<(), String> {
        self.subtract_from(nodal_fields, residual)
    }
}

impl<L> Conservative<NodalFields<L>, Quantity<L::Energy>> for NodalSource<L>
where
    L: Diffusive,
    Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>,
{
    fn potential_energy(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, String> {
        self.check(nodal_fields.len())?;
        Ok(-nodal_fields
            .iter()
            .zip(&self.sources)
            .map(|(&field, &source)| field * source)
            .sum::<Quantity<L::Energy>>())
    }
}

impl<L> ConservativeAt<NodalFields<L>, Quantity<L::Energy>> for NodalSource<L>
where
    L: Diffusive,
    Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>,
{
    fn potential_energy_at(
        &self,
        _: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, String> {
        self.potential_energy(nodal_fields)
    }
}
