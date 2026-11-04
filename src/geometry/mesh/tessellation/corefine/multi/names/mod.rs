//! Naming the points where surfaces meet, by the numbers of all the solids together.

use super::super::{Feature, Key};
use std::collections::HashMap;

/// Where a solid's numbers start in those of all the solids together.
pub(super) struct Offsets {
    pub node: usize,
    pub triangle: usize,
}

fn globalize(feature: &Feature, offsets: &Offsets) -> Feature {
    match feature {
        Feature::Vertex(node) => Feature::Vertex(node + offsets.node),
        Feature::Edge([a, b]) => Feature::Edge([a + offsets.node, b + offsets.node]),
        Feature::Face(triangle) => Feature::Face(triangle + offsets.triangle),
    }
}

/// The key of a point as seen from one of the two surfaces it is on, with that surface's feature
/// first and the other's second, named by the numbers of all the solids together.
pub(super) fn viewed(key: &Key, from_first: bool, offsets: [&Offsets; 2]) -> Key {
    let (first, second) = (
        globalize(&key.first, offsets[0]),
        globalize(&key.second, offsets[1]),
    );
    if from_first {
        Key { first, second }
    } else {
        Key {
            first: second,
            second: first,
        }
    }
}

/// The same point seen from the other surface is the same point.
pub(super) fn canonical(key: &Key) -> Key {
    if key.first <= key.second {
        key.clone()
    } else {
        Key {
            first: key.second.clone(),
            second: key.first.clone(),
        }
    }
}

/// The root of the group of identical nodes that this one is in, joining it to itself if new.
pub(super) fn find(parents: &mut HashMap<usize, usize>, node: usize) -> usize {
    let parent = *parents.entry(node).or_insert(node);
    if parent == node {
        node
    } else {
        let root = find(parents, parent);
        parents.insert(node, root);
        root
    }
}
