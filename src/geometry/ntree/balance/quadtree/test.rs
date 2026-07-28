use crate::geometry::ntree::{
    Quadtree,
    balance::{Balance, Balancing},
    node::{Kind, Node},
    pair::Pairing,
    rescale::Rescaling,
};

fn fuzz_tree(seed: u64) -> Quadtree<u16, usize> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
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
            center: [16.0; 2],
            cell: 1.0,
            half: 16.0,
        },
    };
    quadtree.subdivide(0).unwrap();
    for _ in 0..40 {
        let leaves: Vec<usize> = quadtree
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_leaf() && node.length >= 4)
            .map(|(i, _)| i)
            .collect();
        if leaves.is_empty() {
            break;
        }
        let pick = leaves[rand() % leaves.len()];
        quadtree.subdivide(pick).unwrap();
    }
    quadtree
}

#[test]
fn weak_2_no_finer_than_weak_1() {
    for seed in 0..100 {
        let mut weak_1 = fuzz_tree(seed);
        let mut weak_2 = fuzz_tree(seed);
        weak_1.balance(Balancing::Weak(1));
        weak_2.balance(Balancing::Weak(2));
        assert!(weak_2.nodes.len() <= weak_1.nodes.len(), "seed {seed}");
    }
}

fn build_jump() -> Quadtree<u16, usize> {
    let mut quadtree = Quadtree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0],
            length: 8,
            facets: [None; 4],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: [4.0; 2],
            cell: 1.0,
            half: 4.0,
        },
    };
    quadtree.subdivide(0).unwrap();
    quadtree.subdivide(1).unwrap();
    let orthant = quadtree.nodes[1].orthants().unwrap()[1];
    quadtree.subdivide(orthant).unwrap();
    quadtree
}

#[test]
fn weak_2_accepts_two_level_jump() {
    let build = build_jump;
    let mut weak_1 = build();
    let mut weak_2 = build();
    assert!(!weak_1.balance(Balancing::Weak(1)));
    assert!(weak_2.balance(Balancing::Weak(2)));
    assert_eq!(weak_2.nodes.len(), 13);
    assert!(weak_1.nodes.len() > 13);
}

#[test]
fn strong_2_accepts_two_level_jump() {
    let mut strong_1 = build_jump();
    let mut strong_2 = build_jump();
    assert!(!strong_1.balance(Balancing::Strong(1)));
    assert!(strong_2.balance(Balancing::Strong(2)));
    assert!(strong_1.nodes.len() > strong_2.nodes.len());
}

#[test]
fn strong_is_never_coarser_than_weak() {
    (1..4).for_each(|depth| {
        let mut strong = build_jump();
        let mut weak = build_jump();
        strong.balance(Balancing::Strong(depth));
        weak.balance(Balancing::Weak(depth));
        assert!(
            strong.nodes.len() >= weak.nodes.len(),
            "depth {depth}: strong {} < weak {}",
            strong.nodes.len(),
            weak.nodes.len()
        )
    })
}

#[test]
fn strong_deeper_is_never_finer() {
    let mut previous = usize::MAX;
    (1..4).for_each(|depth| {
        let mut tree = build_jump();
        tree.balance(Balancing::Strong(depth));
        assert!(
            tree.nodes.len() <= previous,
            "depth {depth} refined more than {}",
            depth - 1
        );
        previous = tree.nodes.len()
    })
}
