#[cfg(test)]
mod test;

use crate::geometry::{
    grid::{Pixels, Voxels},
    ntree::{Octree, Quadtree},
};
use std::hash::Hash;

impl<V: Copy + Eq + Hash> Pixels<V> {
    pub fn defeature(self, minimum: usize) -> Self {
        let mut quadtree = Quadtree::<u16, usize, V>::from(self);
        quadtree.defeature(minimum);
        Self::from(&quadtree)
    }
}

impl<V: Copy + Eq + Hash> Voxels<V> {
    pub fn defeature(self, minimum: usize) -> Self {
        let mut octree = Octree::<u16, usize, V>::from(self);
        octree.defeature(minimum);
        Self::from(&octree)
    }
}
