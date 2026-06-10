pub mod elastic;
pub mod elastic_viscoplastic;
pub mod hyperelastic;

use crate::{
    constitutive::solid::Solid,
    fem::{
        Blocks, ElasticViscoplasticAndElastic, Model,
        block::{Block, element::solid::SolidFiniteElement},
    },
    math::TensorRank2Vec2D,
    mechanics::Forces,
};
use std::fmt::Debug;

pub type NodalForcesSolid = Forces;
pub type NodalStiffnessesSolid = TensorRank2Vec2D<3, 1, 1>;

pub trait SolidFiniteElements
where
    Self: Debug,
{
}

impl<B> SolidFiniteElements for Model<B> where B: SolidFiniteElements {}

impl<B1, B2> SolidFiniteElements for Blocks<B1, B2>
where
    B1: SolidFiniteElements,
    B2: SolidFiniteElements,
{
}

impl<B1, B2> SolidFiniteElements for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: SolidFiniteElements,
    B2: SolidFiniteElements,
{
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> SolidFiniteElements
    for Block<C, F, G, M, N, P>
where
    C: Solid,
    F: SolidFiniteElement<G, M, N, P>,
{
}
