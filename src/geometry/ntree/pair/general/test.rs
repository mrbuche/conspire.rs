use crate::geometry::ntree::{
    BinaryTree, Quadtree,
    balance::Balancing,
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
        rescale: Rescaling {
            center: [0.0; 1],
            cell: 1.0,
            half: 0.0,
        },
    };
    binary_tree.subdivide(1).unwrap();
    binary_tree.subdivide(2).unwrap();
    binary_tree.subdivide(3).unwrap();
    // nodes 5..=10 are the length-4 children of nodes 1, 2, 3 (corners
    // 8, 12, 16, 20, 24, 28); subdivide the middle three (12, 16, 20).
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
