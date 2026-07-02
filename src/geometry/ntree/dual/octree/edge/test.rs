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
            value: None,
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
fn write_weak_edge_dual() {
    use super::super::test::verify_dual;
    use crate::{
        geometry::{mesh::Output, ntree::Dualization},
        io::Write,
    };
    let mut octree = weak_edge_tree(Balancing::Weak);
    let mesh = octree.dualize();
    if let Err(error) = verify_dual(&mesh) {
        panic!("weak dual failed verification: {error}");
    }
    mesh.write(Output::Exodus("target/weak_edge.exo")).unwrap();
}

#[test]
fn transition_5_fills_weak_edge_config_only() {
    let hexes = |balancing| {
        let octree = weak_edge_tree(balancing);
        let (center_nodes, mut coordinates, mut node_index, mut connectivity, mut nodes_map) =
            transitions(&octree);
        let filled = connectivity.len();
        super::transition_5::template(
            &octree,
            &center_nodes,
            &mut coordinates,
            &mut connectivity,
            &mut node_index,
            &mut nodes_map,
        );
        connectivity.len() - filled
    };
    assert_eq!(
        hexes(Balancing::Weak),
        23,
        "transition_5 should fill the weak-balanced edge tube"
    );
    assert_eq!(
        hexes(Balancing::Strong),
        0,
        "transition_5 fired on the strong tree (the config should be balanced away)"
    );
}

type Transitions = (
    Vec<usize>,
    Coordinates<D>,
    usize,
    Vec<[usize; N]>,
    NodeMap<D>,
);

fn transitions(octree: &Octree<u16, usize>) -> Transitions {
    use crate::geometry::ntree::dual::Uniform;
    let (center_nodes, mut coordinates, mut node_index, mut connectivity) = octree.initialize();
    octree.uniform_transitions(&center_nodes, &mut connectivity);
    let mut nodes_map = NodeMap::new();
    super::super::face::face_transition(
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
    );
    (
        center_nodes,
        coordinates,
        node_index,
        connectivity,
        nodes_map,
    )
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
