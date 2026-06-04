#[cfg(test)]
mod test;

mod edge_1;
mod edge_2;
mod edge_3;
mod edge_4;
mod face;
mod vertex_1;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh},
        ntree::{
            Octree,
            balance::Balancing,
            dual::{
                Dualization, NodeMap, Uniform,
                octree::{
                    edge_1::edge_transition_1, edge_2::edge_transition_2,
                    edge_3::edge_transition_3, edge_4::edge_transition_4, face::face_transition,
                    vertex_1::vertex_transition_1,
                },
            },
        },
    },
    math::{Scalar, TensorVec},
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

fn get_or_add(
    coordinate: Coordinate<D>,
    coordinates: &mut Coordinates<D>,
    nodes_map: &mut NodeMap<D>,
    node_index: &mut usize,
) -> usize {
    let key = [
        (2.0 * coordinate[0]) as usize,
        (2.0 * coordinate[1]) as usize,
        (2.0 * coordinate[2]) as usize,
    ];
    if let Some(&node) = nodes_map.get(&key) {
        node
    } else {
        let node = *node_index;
        coordinates.push(coordinate);
        nodes_map.insert(key, node);
        *node_index += 1;
        node
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
        face_transition(
            self,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
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
        vertex_transition_1(self, &center_nodes, &mut connectivity);
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
