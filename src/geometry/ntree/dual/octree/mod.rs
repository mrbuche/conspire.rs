#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
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

impl<T, U> Dualization<D> for Octree<T, U>
where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    fn dualize(&mut self) -> Mesh<D> {
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = self.initialize();
        self.uniform_transitions(&center_nodes, &mut connectivity);
        let mut nodes_map = NodeMap::new();
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
        self.rescale_coordinates(&mut coordinates);
        (
            vec![Connectivity::Hexahedral(connectivity.into())],
            coordinates,
        )
            .into()
    }
}

fn face_transition<T, U>(
    _tree: &Octree<T, U>,
    _center_nodes: &[usize],
    _coordinates: &mut Coordinates<D>,
    _connectivity: &mut Vec<[usize; N]>,
    _node_index: &mut usize,
    _nodes_map: &mut NodeMap<D>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
}
