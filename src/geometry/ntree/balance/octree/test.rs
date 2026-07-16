use crate::geometry::ntree::{
    Octree,
    balance::{Balance, Balancing},
    node::{Kind, Node},
    pair::Pairing,
    rescale::Rescaling,
    write::htg::WriteHtg,
};

fn fuzz_tree(seed: u64, length: u16, picks: usize) -> Octree<u16, usize> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut octree = Octree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0, 0],
            length,
            facets: [None; 6],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: [length as f64 / 2.0; 3],
            cell: 1.0,
            half: length as f64 / 2.0,
        },
    };
    octree.subdivide(0).unwrap();
    for _ in 0..picks {
        let leaves: Vec<usize> = octree
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
        octree.subdivide(pick).unwrap();
    }
    octree
}

#[test]
fn weak_2_no_finer_than_weak_1() {
    for seed in 0..100 {
        let mut weak_1 = fuzz_tree(seed, 32, 40);
        let mut weak_2 = fuzz_tree(seed, 32, 40);
        weak_1.balance(Balancing::Weak(1));
        weak_2.balance(Balancing::Weak(2));
        assert!(weak_2.nodes.len() <= weak_1.nodes.len(), "seed {seed}");
    }
}

#[test]
fn write_htg_comparison() {
    for (balancing, path) in [
        (Balancing::Weak(1), "target/balance_weak_1.htg"),
        (Balancing::Weak(2), "target/balance_weak_2.htg"),
        (Balancing::Weak(3), "target/balance_weak_3.htg"),
    ] {
        let mut octree = fuzz_tree(7, 128, 400);
        octree.balance(balancing);
        octree.write_htg(path).unwrap();
    }
}

#[test]
fn weak_2_accepts_two_level_jump() {
    let build = || {
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
                center: [4.0; 3],
                cell: 1.0,
                half: 4.0,
            },
        };
        octree.subdivide(0).unwrap();
        octree.subdivide(1).unwrap();
        let orthant = octree.nodes[1].orthants().unwrap()[1];
        octree.subdivide(orthant).unwrap();
        octree
    };
    let mut weak_1 = build();
    let mut weak_2 = build();
    assert!(!weak_1.balance(Balancing::Weak(1)));
    assert!(weak_2.balance(Balancing::Weak(2)));
    assert_eq!(weak_2.nodes.len(), 25);
    assert!(weak_1.nodes.len() > 25);
}
