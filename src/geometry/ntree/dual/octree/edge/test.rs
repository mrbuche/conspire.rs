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
#[ignore = "writes target/weak_edge.exo for visual inspection"]
fn write_weak_edge_dual() {
    use crate::{
        geometry::{mesh::Output, ntree::Dualization},
        io::Write,
    };
    let mut octree = weak_edge_tree(Balancing::Weak);
    let mesh = octree.dualize();
    mesh.write(Output::Exodus("target/weak_edge.exo")).unwrap();
}

#[test]
fn transition_5_detects_weak_edge_config_only() {
    let weak = weak_edge_tree(Balancing::Weak);
    assert!(
        !super::transition_5::detect(&weak).is_empty(),
        "transition_5 did not detect the (0,1,2,1) config on the weak tree"
    );
    let strong = weak_edge_tree(Balancing::Strong);
    assert!(
        super::transition_5::detect(&strong).is_empty(),
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
