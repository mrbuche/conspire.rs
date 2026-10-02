use crate::{
    fem::{
        Blocks, ElementModel, Elements, Model,
        block::mass::{NodalLumpedMasses, NodalMasses},
    },
    math::Tensor,
};

pub trait ConsistentMassElements
where
    Self: Elements,
{
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses);
}

pub trait LumpedMassElements
where
    Self: Elements,
{
    fn nodal_lumped_masses_into(&self, nodal_lumped_masses: &mut NodalLumpedMasses);
}

impl<B1, B2> ConsistentMassElements for Blocks<B1, B2>
where
    B1: ConsistentMassElements,
    B2: ConsistentMassElements,
{
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses) {
        self.0.nodal_masses_into(nodal_masses);
        self.1.nodal_masses_into(nodal_masses)
    }
}

impl<B1, B2> LumpedMassElements for Blocks<B1, B2>
where
    B1: LumpedMassElements,
    B2: LumpedMassElements,
{
    fn nodal_lumped_masses_into(&self, nodal_lumped_masses: &mut NodalLumpedMasses) {
        self.0.nodal_lumped_masses_into(nodal_lumped_masses);
        self.1.nodal_lumped_masses_into(nodal_lumped_masses)
    }
}

impl<B, const D: usize> ConsistentMassElements for Model<B, D>
where
    B: ConsistentMassElements,
{
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses) {
        self.blocks.nodal_masses_into(nodal_masses)
    }
}

impl<B, const D: usize> LumpedMassElements for Model<B, D>
where
    B: LumpedMassElements,
{
    fn nodal_lumped_masses_into(&self, nodal_lumped_masses: &mut NodalLumpedMasses) {
        self.blocks.nodal_lumped_masses_into(nodal_lumped_masses)
    }
}

impl<B, const D: usize> Model<B, D>
where
    B: ConsistentMassElements,
{
    pub fn nodal_masses(&self) -> NodalMasses {
        let mut nodal_masses = NodalMasses::zero(self.coordinates().len());
        self.nodal_masses_into(&mut nodal_masses);
        nodal_masses
    }
}

impl<B, const D: usize> Model<B, D>
where
    B: LumpedMassElements,
{
    pub fn nodal_lumped_masses(&self) -> NodalLumpedMasses {
        let mut nodal_lumped_masses = NodalLumpedMasses::zero(self.coordinates().len());
        self.nodal_lumped_masses_into(&mut nodal_lumped_masses);
        nodal_lumped_masses
    }
}
