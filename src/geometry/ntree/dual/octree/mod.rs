#[cfg(test)]
mod test;

use crate::{
    geometry::{
        mesh::HexahedralMesh,
        ntree::{
            Octree,
            balance::Balancing,
            dual::{Dualization, Uniform},
        },
    },
    math::Scalar,
};

const D: usize = 3;
const N: usize = 8;

impl<const I: usize, T, U, V> Dualization<D, I, 3, N, V> for Octree<T, U>
where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
    V: Copy + Default + From<usize>,
{
    fn dualize(&mut self) -> HexahedralMesh<I, V> {
        let (center_nodes, coordinates, _node_index, mut connectivity) = self.initialize();
        self.uniform_transitions(&center_nodes, &mut connectivity);
        if matches!(self.balanced, Balancing::Weak) {
            unimplemented!()
        }
        (connectivity, coordinates).into()
    }
}
