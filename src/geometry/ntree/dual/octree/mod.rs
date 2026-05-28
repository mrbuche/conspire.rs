#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::HexahedralMesh,
        ntree::{
            Octree,
            balance::Balancing,
            dual::{Dualization, NodeMap, Uniform},
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
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = self.initialize();
        self.uniform_transitions(&center_nodes, &mut connectivity);
        let mut nodes_map = NodeMap::<D, V>::new();
        face_transition(
            self,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        if matches!(self.balanced, Balancing::Weak) {
            unimplemented!()
        }
        (connectivity, coordinates).into()
    }
}

fn face_transition<const I: usize, T, U, V>(
    _tree: &Octree<T, U>,
    _center_nodes: &[V],
    _coordinates: &mut Coordinates<D, I>,
    _connectivity: &mut Vec<[V; N]>,
    _node_index: &mut usize,
    _nodes_map: &mut NodeMap<D, V>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
    V: Copy + From<usize>,
{
}
