use crate::geometry::ntree::{
    BinaryTree, Quadtree,
    balance::Balancing,
    dual::{Dualization, quadtree::test::verify_dual},
    node::{Kind, Node},
    pair::Pairing,
    rescale::Rescaling,
};

const D: usize = 2;

fn one_quadrant_refined() -> Quadtree<u16, usize> {
    let mut quadtree = Quadtree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0],
            length: 32,
            facets: [None; 4],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        pairing_vertices: Default::default(),
        rescale: Rescaling {
            center: [16.0; D],
            cell: 1.0,
            half: 16.0,
        },
    };
    quadtree.subdivide(0).unwrap();
    let children = *quadtree[0].orthants().unwrap();
    quadtree.subdivide(children[0]).unwrap();
    quadtree
}

#[test]
fn isolated_refinement_needs_no_extra_split() {
    let mut quadtree = one_quadrant_refined();
    let before = quadtree.len();
    let paired = quadtree.pair(Pairing::Generalized).unwrap();
    assert!(paired);
    assert_eq!(quadtree.len(), before);
}

#[test]
fn probe_isolated_refinement_dualizes() {
    let mut quadtree = one_quadrant_refined();
    quadtree.pair(Pairing::Generalized).unwrap();
    quadtree.balanced = Balancing::Weak;
    quadtree.paired = Pairing::Generalized;
    let mesh = quadtree.dualize();
    if let Err(error) = verify_dual(&mesh) {
        panic!("{error}");
    }
}

fn sandwiched_row() -> Quadtree<u16, usize> {
    let mut quadtree = Quadtree::<u16, usize> {
        balanced: Balancing::None,
        nodes: (0..5u16)
            .map(|i| Node {
                corner: [16 * i, 0],
                length: 16,
                facets: [None; 4],
                kind: Kind::Leaf,
                value: None,
            })
            .collect(),
        paired: Pairing::None,
        pairing_vertices: Default::default(),
        rescale: Rescaling {
            center: [0.0; D],
            cell: 1.0,
            half: 0.0,
        },
    };
    quadtree.subdivide(1).unwrap();
    quadtree.subdivide(2).unwrap();
    quadtree.subdivide(3).unwrap();
    quadtree
}

#[test]
fn probe_sandwiched_row_dualizes() {
    let mut quadtree = sandwiched_row();
    let paired = quadtree.pair(Pairing::Generalized).unwrap();
    assert!(!paired);
    assert_ne!(quadtree[0].is_tree(), quadtree[4].is_tree());
    quadtree.balanced = Balancing::Weak;
    quadtree.paired = Pairing::Generalized;
    let mesh = quadtree.dualize();
    if let Err(error) = verify_dual(&mesh) {
        panic!("{error}");
    }
}

#[test]
fn regular_pairing_forces_unnecessary_splits() {
    let mut quadtree = one_quadrant_refined();
    let before = quadtree.len();
    quadtree.pair(Pairing::Regular).unwrap();
    assert!(quadtree.len() > before);
}

fn sandwiched_run() -> BinaryTree<u16, usize> {
    let mut binary_tree = BinaryTree::<u16, usize> {
        balanced: Balancing::None,
        nodes: (0..5u16)
            .map(|i| Node {
                corner: [2 * i],
                length: 2,
                facets: [None; 2],
                kind: Kind::Leaf,
                value: None,
            })
            .collect(),
        paired: Pairing::None,
        pairing_vertices: Default::default(),
        rescale: Rescaling {
            center: [0.0; 1],
            cell: 1.0,
            half: 0.0,
        },
    };
    binary_tree.subdivide(1).unwrap();
    binary_tree.subdivide(2).unwrap();
    binary_tree.subdivide(3).unwrap();
    binary_tree
}

#[test]
fn sandwiched_run_forces_exactly_one_neighbor_split() {
    let mut binary_tree = sandwiched_run();
    let paired = binary_tree.pair(Pairing::Generalized).unwrap();
    assert!(!paired);
    assert_ne!(binary_tree[0].is_tree(), binary_tree[4].is_tree());
}

#[test]
fn sandwiched_run_records_its_pairing_vertices() {
    let mut binary_tree = sandwiched_run();
    binary_tree.pair(Pairing::Generalized).unwrap();
    // hand-verified optimum at coarse=2: {vertex1, vertex3} in lattice
    // units (valence 2 each, cost 4), i.e. absolute corners {2, 6}.
    // vertex0/vertex4 only touch the unrequired end cells, so they cover
    // nothing required and can't participate in a cheaper solution.
    assert_eq!(
        binary_tree.pairing_vertices,
        [[2], [6]].into_iter().collect()
    );
    assert!(binary_tree[0].is_tree());
    assert!(binary_tree[4].is_leaf());
}

fn nested_sandwiched_runs() -> BinaryTree<u16, usize> {
    let mut binary_tree = BinaryTree::<u16, usize> {
        balanced: Balancing::None,
        nodes: (0..5u16)
            .map(|i| Node {
                corner: [8 * i],
                length: 8,
                facets: [None; 2],
                kind: Kind::Leaf,
                value: None,
            })
            .collect(),
        paired: Pairing::None,
        pairing_vertices: Default::default(),
        rescale: Rescaling {
            center: [0.0; 1],
            cell: 1.0,
            half: 0.0,
        },
    };
    binary_tree.subdivide(1).unwrap();
    binary_tree.subdivide(2).unwrap();
    binary_tree.subdivide(3).unwrap();
    binary_tree.subdivide(6).unwrap();
    binary_tree.subdivide(7).unwrap();
    binary_tree.subdivide(8).unwrap();
    binary_tree
}

#[test]
fn generalized_pairing_resolves_every_level_pair_in_one_pass() {
    let mut binary_tree = nested_sandwiched_runs();
    let paired = binary_tree.pair(Pairing::Generalized).unwrap();
    assert!(!paired);
    assert_ne!(binary_tree[0].is_tree(), binary_tree[4].is_tree());
    assert!(binary_tree[5].is_tree() || binary_tree[9].is_tree() || binary_tree[10].is_tree());

    let before = binary_tree.len();
    let paired_again = binary_tree.pair(Pairing::Generalized).unwrap();
    assert!(paired_again);
    assert_eq!(binary_tree.len(), before);
}
