#[cfg(test)]
mod test;

mod edge_1;
mod edge_2;
mod edge_3;
mod edge_4;
mod face;

use crate::{
    geometry::{
        Coordinate,
        mesh::{Connectivity, Mesh},
        ntree::{
            Octree,
            balance::Balancing,
            dual::{
                Dualization, NodeMap, Uniform,
                octree::{
                    edge_1::edge_transition_1, edge_2::edge_transition_2,
                    edge_3::edge_transition_3, edge_4::edge_transition_4, face::face_transition,
                },
            },
        },
    },
    math::Scalar,
};

const D: usize = 3;
const L: usize = 4;
const M: usize = 6;
const N: usize = 8;

const fn facet_direction(facet: usize) -> Coordinate<D> {
    match facet {
        0 => Coordinate::const_from([-1.0, 0.0, 0.0]),
        1 => Coordinate::const_from([1.0, 0.0, 0.0]),
        2 => Coordinate::const_from([0.0, -1.0, 0.0]),
        3 => Coordinate::const_from([0.0, 1.0, 0.0]),
        4 => Coordinate::const_from([0.0, 0.0, -1.0]),
        5 => Coordinate::const_from([0.0, 0.0, 1.0]),
        _ => panic!(),
    }
}

impl<T, U> Dualization<D> for Octree<T, U>
where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    fn dualize(&mut self) -> Mesh<D> {
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = self.initialize();
        self.uniform_transitions(&center_nodes, &mut connectivity);
        let mut nodes_map = NodeMap::new();
        //
        // Could move face back here and use get_or_add in edge transitions.
        //
        edge_transition_1(
            self,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        edge_transition_3(
            self,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        face_transition(
            self,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        edge_transition_2(
            self,
            &center_nodes,
            &coordinates,
            &mut connectivity,
            &nodes_map,
        );
        edge_transition_4(
            self,
            &center_nodes,
            &coordinates,
            &mut connectivity,
            &nodes_map,
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
