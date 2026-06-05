#[cfg(test)]
mod test;

mod edge_1;
mod edge_2;
mod edge_3;
mod edge_4;
mod face;
mod vertex;

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
                    vertex::vertex_transitions,
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

/// Runs the generic "descend toward V" dual on an equilibrated octree and checks it
/// against the cell-center hexes of the template pipeline: every uniform/vt2..21 hex must
/// be a generic vertex star, and no vt1 hex may be (vt1 is a coarse-face filler). Returns
/// `(generic hexes, distinct generic stars, non-vt1 cell-center hexes, vt1 hexes)`.
#[cfg(test)]
pub(crate) fn generic_star_report(octree: &Octree<u16, usize>) -> (usize, usize, usize, usize) {
    use crate::geometry::ntree::dual::{
        Uniform,
        octree::vertex::{generic::vertex_dual_generic, transition_1_only, vertex_transitions},
    };
    use std::collections::HashSet;
    let (center_nodes, coordinates, ..) = octree.initialize();
    let as_sets = |hexes: &[[usize; N]]| -> HashSet<[usize; N]> {
        hexes
            .iter()
            .map(|hex| {
                let mut sorted = *hex;
                sorted.sort_unstable();
                sorted
            })
            .collect()
    };
    let generic = vertex_dual_generic(octree, &center_nodes);
    generic.iter().enumerate().for_each(|(i, hex)| {
        let p: [[Scalar; D]; N] = std::array::from_fn(|k| {
            let v = &coordinates[hex[k]];
            std::array::from_fn(|a| v[a])
        });
        let tet = |a: usize, b: usize, c: usize, d: usize| -> Scalar {
            let ab: [Scalar; D] = std::array::from_fn(|x| p[b][x] - p[a][x]);
            let ac: [Scalar; D] = std::array::from_fn(|x| p[c][x] - p[a][x]);
            let ad: [Scalar; D] = std::array::from_fn(|x| p[d][x] - p[a][x]);
            ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
                + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
        };
        let vol6 = tet(0, 1, 2, 6)
            + tet(0, 2, 3, 6)
            + tet(0, 3, 7, 6)
            + tet(0, 7, 4, 6)
            + tet(0, 4, 5, 6)
            + tet(0, 5, 1, 6);
        assert!(
            vol6 > 1e-12,
            "generic hex {i} not positively oriented: {hex:?}"
        );
    });
    let generic_sets = as_sets(&generic);
    let mut stars = Vec::new();
    octree.uniform_transitions(&center_nodes, &mut stars);
    vertex_transitions(octree, &center_nodes, &mut stars);
    let vt1 = transition_1_only(octree, &center_nodes);
    let vt1_sets = as_sets(&vt1);
    let non_vt1: HashSet<_> = as_sets(&stars).difference(&vt1_sets).copied().collect();
    let absent = non_vt1.difference(&generic_sets).count();
    assert_eq!(
        absent, 0,
        "{absent} non-vt1 cell-center hexes absent from generic dual"
    );
    let vt1_in_generic = vt1_sets.intersection(&generic_sets).count();
    assert_eq!(
        vt1_in_generic, 0,
        "{vt1_in_generic} vt1 hexes were vertex stars"
    );
    (generic.len(), generic_sets.len(), non_vt1.len(), vt1.len())
}

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
        vertex_transitions(self, &center_nodes, &mut connectivity);
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
