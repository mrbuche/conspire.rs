//! Naming the points where surfaces meet, by the numbers of all the solids together.

#[cfg(test)]
mod test;

use super::super::{Feature, Key};
use crate::math::FxHashMap;
use core::{hash::Hash, iter::once};

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

/// A point as seen from one surface it is on: the lowest feature of that surface it is on, and the
/// lowest feature of every other surface it is on, in order, all by the numbers of all the solids
/// together.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct Spot {
    pub own: Feature,
    pub others: Vec<Feature>,
}

impl From<Key> for Spot {
    fn from(key: Key) -> Self {
        Self {
            own: key.first,
            others: vec![key.second],
        }
    }
}

/// The name of a point: the lowest feature of every solid whose surface it is on, in order.
///
/// However many surfaces meet at the point, and in whichever order they were compared, the same
/// point has the same name.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct Name(Vec<Feature>);

impl Name {
    pub fn new(features: impl IntoIterator<Item = Feature>) -> Self {
        let mut features: Vec<Feature> = features.into_iter().collect();
        features.sort_unstable();
        features.dedup();
        Self(features)
    }
}

impl Name {
    pub fn features(&self) -> &[Feature] {
        &self.0
    }
}

/// The solid that a feature is of, by the numbers of all the solids together.
pub(super) fn solid_of(feature: &Feature, offsets: &[Offsets]) -> usize {
    let (node, triangle) = match feature {
        Feature::Vertex(node) | Feature::Edge([node, _]) => (Some(*node), None),
        Feature::Face(triangle) => (None, Some(*triangle)),
    };
    offsets
        .iter()
        .rposition(|offset| {
            node.is_some_and(|n| offset.node <= n) || triangle.is_some_and(|t| offset.triangle <= t)
        })
        .unwrap_or(0)
}

impl From<&Spot> for Name {
    fn from(spot: &Spot) -> Self {
        Self::new(once(spot.own.clone()).chain(spot.others.iter().cloned()))
    }
}

impl From<&Key> for Name {
    fn from(key: &Key) -> Self {
        Self::new([key.first.clone(), key.second.clone()])
    }
}

/// Groups of items known to be the same, each with the smallest of them as its root.
pub(super) struct Groups<T> {
    parents: FxHashMap<T, T>,
}

impl<T> Default for Groups<T> {
    fn default() -> Self {
        Self {
            parents: FxHashMap::default(),
        }
    }
}

impl<T: Clone + Eq + Hash + Ord> Groups<T> {
    /// The root of the group that this item is in, which is itself if it is in none.
    pub fn find(&mut self, item: &T) -> T {
        let parent = self
            .parents
            .get(item)
            .cloned()
            .unwrap_or_else(|| item.clone());
        if &parent == item {
            return parent;
        }
        let root = self.find(&parent);
        self.parents.insert(item.clone(), root.clone());
        root
    }
    /// Says that two items are the same, joining their groups.
    pub fn join(&mut self, a: &T, b: &T) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            let (low, high) = if a < b { (a, b) } else { (b, a) };
            self.parents.insert(high, low);
        }
    }
    /// The root of the group of every item that is in a group of more than itself.
    pub fn roots(&mut self) -> FxHashMap<T, T> {
        let items: Vec<T> = self.parents.keys().cloned().collect();
        items
            .into_iter()
            .map(|item| {
                let root = self.find(&item);
                (item, root)
            })
            .collect()
    }
}
