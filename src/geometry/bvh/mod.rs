mod base;
mod from;
mod node;
mod primitive;
mod ray;

use crate::{geometry::bvh::node::Nodes, math::Scalar};

pub struct BoundingVolumeHierarchy<const D: usize> {
    items: Vec<usize>,
    nodes: Nodes<D>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    distance: Scalar,
    index: usize,
}

impl Hit {
    pub fn distance(&self) -> Scalar {
        self.distance
    }
    pub fn index(&self) -> usize {
        self.index
    }
}
