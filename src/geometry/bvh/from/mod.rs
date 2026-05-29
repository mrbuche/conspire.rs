#[cfg(test)]
mod test;

use crate::geometry::{
    bvh::{BoundingVolumeHierarchy, primitive::Primitives},
    mesh::{Mesh, tessellation::Tessellation},
};

impl<const D: usize, T> From<Primitives<D, T>> for BoundingVolumeHierarchy<D, T>
where
    T: Copy,
{
    fn from(mut primitives: Primitives<D, T>) -> Self {
        let mut bvh = Self {
            items: Vec::new(),
            nodes: Vec::new(),
        };
        let leaf_size = 4;
        bvh.build_node(&mut primitives, leaf_size);
        bvh
    }
}

impl<const D: usize, T, U, V> From<&Mesh<D, T>> for BoundingVolumeHierarchy<D, V>
where
    for<'a> &'a T: IntoIterator<Item = &'a U>,
    for<'a> &'a U: IntoIterator<Item = &'a V>,
    V: Copy + From<usize> + Into<usize>,
{
    fn from(mesh: &Mesh<D, T>) -> Self {
        Primitives::from(mesh).into()
    }
}

impl<V> From<&Tessellation<V>> for BoundingVolumeHierarchy<3, V>
where
    for<'a> &'a [V; 3]: IntoIterator<Item = &'a V>,
    V: Copy + From<usize> + Into<usize>,
{
    fn from(tessellation: &Tessellation<V>) -> Self {
        BoundingVolumeHierarchy::<3, V>::from(tessellation.mesh())
    }
}
