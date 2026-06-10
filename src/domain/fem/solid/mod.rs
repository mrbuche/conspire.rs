pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::{
    fem::{Blocks, ElasticViscoplasticAndElastic, FiniteElements, Model},
    math::TensorRank2Vec2D,
    mechanics::Forces,
};

pub type NodalForcesSolid = Forces;
pub type NodalStiffnessesSolid = TensorRank2Vec2D<3, 1, 1>;

pub trait SolidFiniteElements
where
    Self: FiniteElements,
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
