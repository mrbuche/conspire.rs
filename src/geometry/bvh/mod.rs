use crate::units::Length;
mod base;
mod from;
mod node;
mod primitive;
mod ray;

use crate::{geometry::bvh::node::Nodes, math::Quantity};

pub struct BoundingVolumeHierarchy<const D: usize> {
    items: Vec<usize>,
    nodes: Nodes<D>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    distance: Quantity<Length>,
    index: usize,
}

impl Hit {
    pub fn distance(&self) -> Quantity<Length> {
        self.distance
    }
    pub fn index(&self) -> usize {
        self.index
    }
}
