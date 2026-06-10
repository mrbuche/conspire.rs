use crate::geometry::{
    Coordinates,
    ntree::{Octree, balance::Balancing, dual::Uniform, pair::Pairing},
};
use std::collections::HashSet;

fn hex_vol6(hex: &[usize; 8], coordinates: &Coordinates<3>) -> f64 {
    let p: [[f64; 3]; 8] = std::array::from_fn(|k| {
        let v = &coordinates[hex[k]];
        [v[0], v[1], v[2]]
    });
    let tet = |a: usize, b: usize, c: usize, d: usize| -> f64 {
        let ab = [p[b][0] - p[a][0], p[b][1] - p[a][1], p[b][2] - p[a][2]];
        let ac = [p[c][0] - p[a][0], p[c][1] - p[a][1], p[c][2] - p[a][2]];
        let ad = [p[d][0] - p[a][0], p[d][1] - p[a][1], p[d][2] - p[a][2]];
        ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
            + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
    };
    tet(0, 1, 2, 6)
        + tet(0, 2, 3, 6)
        + tet(0, 3, 7, 6)
        + tet(0, 7, 4, 6)
        + tet(0, 4, 5, 6)
        + tet(0, 5, 1, 6)
}

#[test]
fn vt21_fires_on_synthetic_checkerboard() {
    use super::vertex::test::{transition_21_only, vertex_dual_generic};
    use crate::geometry::ntree::{
        node::{Kind, Node},
        rescale::Rescaling,
    };

    let mut octree = Octree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0, 0],
            length: 8,
            facets: [None; 6],
            kind: Kind::Leaf,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: [4.0, 4.0, 4.0],
            cell: 1.0,
            half: 4.0,
        },
    };
    octree.subdivide(0).unwrap();
    for macro_cell in 1..=8usize {
        octree.subdivide(macro_cell).unwrap();
    }
    for fine_macro in [1usize, 4, 6, 7] {
        let children = *octree.nodes[fine_macro].orthants().unwrap();
        for child in children {
            octree.subdivide(child).unwrap();
        }
    }
    octree.balanced = Balancing::Strong;
    octree.paired = Pairing::Regular;

    let (center_nodes, coordinates, ..) = octree.initialize();
    let hexes = transition_21_only(&octree, &center_nodes);
    assert!(
        !hexes.is_empty(),
        "vt21 did not fire on the checkerboard vertex"
    );

    hexes.iter().enumerate().for_each(|(i, hex)| {
        assert!(
            hex_vol6(hex, &coordinates) > 1e-12,
            "vt21 hex {i} not positively oriented: {hex:?}"
        );
    });

    let generic: HashSet<[usize; 8]> = vertex_dual_generic(&octree, &center_nodes)
        .into_iter()
        .map(|mut hex| {
            hex.sort_unstable();
            hex
        })
        .collect();
    hexes.iter().for_each(|hex| {
        let mut sorted = *hex;
        sorted.sort_unstable();
        assert!(
            generic.contains(&sorted),
            "vt21 hex {hex:?} is not a vertex star"
        );
    });
}
