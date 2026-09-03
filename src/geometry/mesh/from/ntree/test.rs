use crate::{
    geometry::{
        Coordinate,
        ntree::{
            Balancing, Octree, Pairing, Quadtree, Rescaling,
            node::{Kind, Node},
        },
    },
    math::Quantity,
};

pub(super) fn octree(length: u16) -> Octree<u16, usize> {
    Octree {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0; 3],
            length,
            facets: [None; 6],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: Coordinate::const_from([length as f64 / 2.0; 3]),
            cell: Quantity::new(1.0),
            half: length as f64 / 2.0,
        },
    }
}

pub(super) fn quadtree(length: u16) -> Quadtree<u16, usize> {
    Quadtree {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0; 2],
            length,
            facets: [None; 4],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: Coordinate::const_from([length as f64 / 2.0; 2]),
            cell: Quantity::new(1.0),
            half: length as f64 / 2.0,
        },
    }
}
