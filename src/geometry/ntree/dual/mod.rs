pub mod quadtree;
// pub mod octree;

use crate::geometry::PrimitiveMesh;

pub trait Dualization<const D: usize, const I: usize, const M: usize, const N: usize, T> {
    fn dualize(&mut self) -> PrimitiveMesh<D, I, M, N, T>;
}
