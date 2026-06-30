use crate::geometry::{
    Coordinates,
    mesh::Mesh,
    ntree::{Balance, Dualization, Octree, balance::Balancing, dual::Uniform, pair::Pairing},
};
use std::collections::{HashMap, HashSet};

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

const HEX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

fn verify_dual(mesh: &Mesh<3>) -> Result<(), String> {
    let coordinates = mesh.coordinates();
    for (e, element) in mesh.iter().flatten().enumerate() {
        let mut distinct = element.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != 8 {
            return Err(format!("hex {e} has repeated nodes: {element:?}"));
        }
        let hex: [usize; 8] = std::array::from_fn(|k| element[k]);
        let six_v = hex_vol6(&hex, coordinates);
        if six_v <= 1e-9 {
            return Err(format!(
                "hex {e} not positively oriented (6V={six_v}): {element:?}"
            ));
        }
    }
    let mut faces: HashMap<[usize; 4], usize> = HashMap::new();
    for element in mesh.iter().flatten() {
        for face in HEX_FACES {
            let mut key = [
                element[face[0]],
                element[face[1]],
                element[face[2]],
                element[face[3]],
            ];
            key.sort_unstable();
            *faces.entry(key).or_insert(0) += 1;
        }
    }
    if let Some((face, count)) = faces.iter().find(|(_, count)| **count > 2) {
        return Err(format!("non-conformal: face {face:?} shared {count} times"));
    }
    let mut edges: HashMap<[usize; 2], usize> = HashMap::new();
    for face in mesh.exterior_faces() {
        for i in 0..face.len() {
            let mut edge = [face[i], face[(i + 1) % face.len()]];
            edge.sort_unstable();
            *edges.entry(edge).or_insert(0) += 1;
        }
    }
    if let Some((edge, count)) = edges.iter().find(|(_, count)| **count != 2) {
        return Err(format!(
            "boundary not a closed manifold: edge {edge:?} borders {count} boundary faces"
        ));
    }
    Ok(())
}

fn refine_sets() -> Vec<Vec<usize>> {
    vec![vec![1], vec![1, 2, 3], vec![1, 8], vec![1, 4, 6, 7]]
}

#[test]
fn dual_is_conformal_on_strong_trees() {
    for macros in refine_sets() {
        let mut octree = tree_refine_macros(&macros);
        let mesh = octree.dualize();
        assert!(
            mesh.iter().flatten().count() > 0,
            "dual produced no hexes for {macros:?}"
        );
        if let Err(error) = verify_dual(&mesh) {
            panic!("strong dual {macros:?} failed verification: {error}");
        }
    }
}

fn tree_refine_macros(fine_macros: &[usize]) -> Octree<u16, usize> {
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
            value: None,
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
    for &fine_macro in fine_macros {
        let children = *octree.nodes[fine_macro].orthants().unwrap();
        for child in children {
            octree.subdivide(child).unwrap();
        }
    }
    octree
        .equilibrate(Balancing::Strong, Pairing::Regular)
        .unwrap();
    octree
}

fn edge_counts(octree: &Octree<u16, usize>) -> [usize; 4] {
    use super::{edge::test::edge_transition_counts, face::face_transition};
    let (center_nodes, mut coordinates, mut node_index, mut connectivity) = octree.initialize();
    octree.uniform_transitions(&center_nodes, &mut connectivity);
    let mut nodes_map = HashMap::new();
    face_transition(
        octree,
        &center_nodes,
        &mut coordinates,
        &mut connectivity,
        &mut node_index,
        &mut nodes_map,
    );
    edge_transition_counts(
        octree,
        &center_nodes,
        &mut coordinates,
        &mut connectivity,
        &mut node_index,
        &mut nodes_map,
    )
}

#[test]
fn edge_transitions_each_fire_across_strong_trees() {
    let mut fired = [false; 4];
    for macros in refine_sets() {
        let counts = edge_counts(&tree_refine_macros(&macros));
        (0..4).for_each(|k| fired[k] |= counts[k] > 0);
    }
    for (i, &f) in fired.iter().enumerate() {
        assert!(
            f,
            "edge transition_{} never fired across any test tree",
            i + 1
        );
    }
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
            value: None,
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
