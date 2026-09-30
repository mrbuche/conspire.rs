pub mod solid;

use crate::{
    math::{Quantity, Reference, TensorRank1},
    units::{ReciprocalLength, Volume},
};

pub(crate) type GradientVector = TensorRank1<3, Reference, ReciprocalLength>;

#[derive(Clone, Debug)]
pub(crate) struct Point {
    weight: Quantity<Volume>,
    neighbors: Vec<usize>,
    gradient_vectors: Vec<GradientVector>,
}

impl Point {
    pub(crate) fn new(
        weight: Quantity<Volume>,
        neighbors: Vec<usize>,
        gradient_vectors: Vec<GradientVector>,
    ) -> Self {
        Self {
            weight,
            neighbors,
            gradient_vectors,
        }
    }
    pub(crate) fn neighbors(&self) -> &[usize] {
        &self.neighbors
    }
    pub(crate) fn gradient_vectors(&self) -> &[GradientVector] {
        &self.gradient_vectors
    }
    pub(crate) fn weight(&self) -> &Quantity<Volume> {
        &self.weight
    }
}
