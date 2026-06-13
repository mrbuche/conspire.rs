use super::{D, N};
use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Balance, Octree,
            balance::Balancing,
            dual::NodeMap,
            node::{Kind, Node},
            pair::Pairing,
            rescale::Rescaling,
        },
    },
    math::Scalar,
};

fn refine_to(octree: &mut Octree<u16, usize>, node: usize, levels: usize) {
    if levels == 0 {
        return;
    }
    octree.subdivide(node).unwrap();
    for child in *octree.nodes[node].orthants().unwrap() {
        refine_to(octree, child, levels - 1);
    }
}

fn weak_edge_tree(balancing: Balancing) -> Octree<u16, usize> {
    let mut octree = Octree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0, 0],
            length: 16,
            facets: [None; 6],
            kind: Kind::Leaf,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: [8.0, 8.0, 8.0],
            cell: 1.0,
            half: 8.0,
        },
    };
    octree.subdivide(0).unwrap();
    let macros = *octree.nodes[0].orthants().unwrap();
    let depths = [1usize, 2, 2, 3, 1, 2, 2, 3];
    for (orthant, &levels) in depths.iter().enumerate() {
        refine_to(&mut octree, macros[orthant], levels);
    }
    octree.equilibrate(balancing, Pairing::Regular).unwrap();
    octree
}

#[test]
// #[ignore = "writes target/weak_edge.exo for visual inspection"]
fn write_weak_edge_dual() {
    use crate::{
        geometry::{mesh::Output, ntree::Dualization},
        io::Write,
    };
    let mut octree = weak_edge_tree(Balancing::Weak);
    let mesh = octree.dualize();
    let coordinates = mesh.coordinates();
    let vol6 = |hex: &[usize]| {
        let p: [[f64; 3]; 8] =
            std::array::from_fn(|k| std::array::from_fn(|i| coordinates[hex[k]][i]));
        let tet = |a: usize, b: usize, c: usize, d: usize| {
            let e = |i: usize, j: usize| [p[j][0] - p[i][0], p[j][1] - p[i][1], p[j][2] - p[i][2]];
            let (u, v, w) = (e(a, b), e(a, c), e(a, d));
            u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                + u[2] * (v[0] * w[1] - v[1] * w[0])
        };
        tet(0, 1, 2, 6) + tet(0, 2, 3, 6) + tet(0, 3, 7, 6) + tet(0, 7, 4, 6)
            + tet(0, 4, 5, 6) + tet(0, 5, 1, 6)
    };
    let inverted = mesh.iter().flatten().filter(|hex| vol6(hex) <= 1e-9).count();
    assert_eq!(inverted, 0, "{inverted} non-positive hexes in weak dual");
    mesh.write(Output::Exodus("target/weak_edge.exo")).unwrap();
}

#[test]
fn transition_5_fires_on_weak_edge_config_only() {
    use crate::geometry::ntree::dual::Uniform;
    use std::collections::HashMap;
    let pushes = |balancing| {
        let octree = weak_edge_tree(balancing);
        let (center_nodes, mut coordinates, mut node_index, mut connectivity) = octree.initialize();
        let mut nodes_map = HashMap::new();
        super::transition_5::template(
            &octree,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        connectivity.len()
    };
    assert!(
        pushes(Balancing::Weak) > 0,
        "transition_5 did not fire on the weak tree"
    );
    assert_eq!(
        pushes(Balancing::Strong),
        0,
        "transition_5 fired on the strong tree (the config should be balanced away)"
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn edge_transition_counts<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    coordinates: &mut Coordinates<D>,
    connectivity: &mut Vec<[usize; N]>,
    node_index: &mut usize,
    nodes_map: &mut NodeMap<D>,
) -> [usize; 4]
where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    let mut counts = [0usize; 4];
    let mut len = connectivity.len();
    super::transition_1::template(
        tree,
        center_nodes,
        coordinates,
        connectivity,
        node_index,
        nodes_map,
    );
    counts[0] = connectivity.len() - len;
    len = connectivity.len();
    super::transition_3::template(
        tree,
        center_nodes,
        coordinates,
        connectivity,
        node_index,
        nodes_map,
    );
    counts[2] = connectivity.len() - len;
    len = connectivity.len();
    super::transition_2::template(tree, center_nodes, coordinates, connectivity, nodes_map);
    counts[1] = connectivity.len() - len;
    len = connectivity.len();
    super::transition_4::template(tree, center_nodes, coordinates, connectivity, nodes_map);
    counts[3] = connectivity.len() - len;
    counts
}
