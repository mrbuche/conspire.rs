pub mod elastic;

use crate::{
    constitutive::solid::Solid,
    fem::{
        Blocks, Model,
        block::{Block, element::solid::SolidFiniteElement},
    },
};
use std::fmt::Debug;

pub trait SolidFiniteElementModel
where
    Self: Debug,
{
}

impl<B> SolidFiniteElementModel for Model<B> where B: SolidFiniteElementModel {}

impl<B1, B2> SolidFiniteElementModel for Blocks<B1, B2>
where
    B1: SolidFiniteElementModel,
    B2: SolidFiniteElementModel,
{
}

//
// Might need Block to depend on G and M.
//
impl<C, F, const G: usize, const N: usize> SolidFiniteElementModel for Block<C, F, G, N>
where
    C: Solid,
    F: SolidFiniteElement<G, 3, N>,
{
}
