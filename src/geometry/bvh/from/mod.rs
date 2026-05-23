#[cfg(test)]
mod test;

use crate::geometry::{
    bvh::{BoundingVolumeHierarchy, primitive::Primitives},
    mesh::{Mesh, tessellation::Tessellation},
};

impl<const D: usize, const I: usize, T> From<Primitives<D, I, T>>
    for BoundingVolumeHierarchy<D, I, T>
where
    T: Copy,
{
    fn from(mut primitives: Primitives<D, I, T>) -> Self {
        let mut bvh = Self {
            items: Vec::new(),
            nodes: Vec::new(),
        };
        let leaf_size = 4;
        bvh.build_node(&mut primitives, leaf_size);
        bvh
    }
}

impl<const D: usize, const I: usize, const M: usize, T, U, V> From<&Mesh<D, I, M, T>>
    for BoundingVolumeHierarchy<D, I, V>
where
    for<'a> &'a T: IntoIterator<Item = &'a U>,
    for<'a> &'a U: IntoIterator<Item = &'a V>,
    V: Copy + From<usize> + Into<usize>,
{
    fn from(mesh: &Mesh<D, I, M, T>) -> Self {
        Primitives::from(mesh).into()
    }
}

impl<const I: usize, V> From<&Tessellation<I, V>> for BoundingVolumeHierarchy<3, I, V>
where
    for<'a> &'a [V; 3]: IntoIterator<Item = &'a V>,
    V: Copy + From<usize> + Into<usize>,
{
    fn from(tessellation: &Tessellation<I, V>) -> Self {
        BoundingVolumeHierarchy::<3, I, V>::from(tessellation.mesh())
    }
}
