#[cfg(test)]
mod test;

use crate::geometry::{
    grid::{Pixels, Voxels},
    ntree::{Octree, Quadtree},
};
use std::hash::Hash;

impl<V: Copy + Eq + Hash> Pixels<V> {
    pub fn defeature(self, minimum: usize) -> Result<Self, &'static str> {
        let mut quadtree = Quadtree::<u16, usize, V>::try_from(self)?;
        quadtree.defeature(minimum);
        Ok(Self::from(&quadtree))
    }
}

impl<V: Copy + Eq + Hash> Voxels<V> {
    pub fn defeature(self, minimum: usize) -> Result<Self, &'static str> {
        let mut octree = Octree::<u16, usize, V>::try_from(self)?;
        octree.defeature(minimum);
        Ok(Self::from(&octree))
    }
}
