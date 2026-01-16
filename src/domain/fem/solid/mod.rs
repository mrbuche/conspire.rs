pub mod elastic;

use crate::{
    constitutive::solid::Solid,
    fem::{
        Blocks, Model,
        block::{Block, element::solid::SolidFiniteElement},
    },
    math::TensorRank2Vec2D,
    mechanics::Forces,
};
use std::fmt::Debug;

pub type NodalForcesSolid = Forces;
pub type NodalStiffnessesSolid = TensorRank2Vec2D<3, 1, 1>;

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

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> SolidFiniteElementModel
    for Block<C, F, G, M, N, P>
where
    C: Solid,
    F: SolidFiniteElement<G, M, N, P>,
{
}
