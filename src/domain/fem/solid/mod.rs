pub mod elastic;
pub mod elastic_viscoplastic;
pub mod hyperelastic;

use crate::{
    fem::{Blocks, ElasticViscoplasticAndElastic, Model},
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
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]);
}

impl<B> SolidFiniteElements for Model<B>
where
    B: SolidFiniteElements,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.blocks.node_neighbors(neighbors)
    }
}

impl<B1, B2> SolidFiniteElements for Blocks<B1, B2>
where
    B1: SolidFiniteElements,
    B2: SolidFiniteElements,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.0.node_neighbors(neighbors);
        self.1.node_neighbors(neighbors)
    }
}

impl<B1, B2> SolidFiniteElements for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: SolidFiniteElements,
    B2: SolidFiniteElements,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.0.node_neighbors(neighbors);
        self.1.node_neighbors(neighbors)
    }
}
