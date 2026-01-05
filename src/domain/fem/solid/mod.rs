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

impl<C, F, const G: usize, const M: usize, const N: usize> SolidFiniteElementModel
    for Block<C, F, G, M, N>
where
    C: Solid,
    F: SolidFiniteElement<G, M, N>,
{
}
