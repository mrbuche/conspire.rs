pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::{
    fem::{Blocks, ElasticViscoplasticAndElastic, FiniteElements, Model},
    math::{TensorRank1Vec, TensorRank2Vec2D},
};

pub type NodalForcesSolid<const D: usize> = TensorRank1Vec<D, 1>;
pub type NodalStiffnessesSolid<const D: usize> = TensorRank2Vec2D<D, 1, 1>;

pub trait SolidFiniteElements
where
    Self: FiniteElements,
{
}

impl<B, const D: usize> SolidFiniteElements for Model<B, D> where B: SolidFiniteElements {}

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
