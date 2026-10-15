#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, mesh::Mesh},
    math::Tensor,
};
use std::slice::Iter;

macro_rules! sets {
    ($set: ident, $sets: ident, $member: ty, $members: ident, $doc: literal, $docs: literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $set {
            label: Option<usize>,
            name: Option<String>,
            $members: Vec<$member>,
        }

        impl $set {
            pub fn new($members: Vec<$member>) -> Self {
                Self {
                    label: None,
                    name: None,
                    $members,
                }
            }
            pub fn with_label(mut self, label: usize) -> Self {
                self.label = Some(label);
                self
            }
            pub fn with_name(mut self, name: impl Into<String>) -> Self {
                self.name = Some(name.into());
                self
            }
            pub fn label(&self) -> Option<usize> {
                self.label
            }
            pub fn name(&self) -> Option<&str> {
                self.name.as_deref()
            }
            pub fn $members(&self) -> &[$member] {
                &self.$members
            }
            /// The set with its members mapped through `map`, dropping those it maps to nothing,
            /// or nothing if no member remains.
            pub fn restrict(&self, map: impl Fn($member) -> Option<$member>) -> Option<Self> {
                let $members: Vec<$member> = self.$members.iter().filter_map(|&m| map(m)).collect();
                (!$members.is_empty()).then(|| Self {
                    label: self.label,
                    name: self.name.clone(),
                    $members,
                })
            }
        }

        #[doc = $docs]
        #[derive(Clone, Debug, Default, PartialEq)]
        pub struct $sets(Vec<$set>);

        impl $sets {
            pub fn new() -> Self {
                Self::default()
            }
            pub fn push(&mut self, set: $set) {
                self.0.push(set)
            }
            pub fn iter(&self) -> Iter<'_, $set> {
                self.0.iter()
            }
            pub fn len(&self) -> usize {
                self.0.len()
            }
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
            /// The first set with a label.
            pub fn by_label(&self, label: usize) -> Option<&$set> {
                self.0.iter().find(|set| set.label == Some(label))
            }
            /// The first set with a name.
            pub fn by_name(&self, name: &str) -> Option<&$set> {
                self.0.iter().find(|set| set.name.as_deref() == Some(name))
            }
            /// The label of each set, giving those without one the smallest label no set has.
            pub fn resolved_labels(&self) -> Vec<usize> {
                let mut next = 0;
                self.0
                    .iter()
                    .map(|set| {
                        set.label.unwrap_or_else(|| {
                            loop {
                                next += 1;
                                if self.by_label(next).is_none() {
                                    return next;
                                }
                            }
                        })
                    })
                    .collect()
            }
        }

        impl<'a> IntoIterator for &'a $sets {
            type Item = &'a $set;
            type IntoIter = Iter<'a, $set>;
            fn into_iter(self) -> Self::IntoIter {
                self.0.iter()
            }
        }

        impl From<Vec<$set>> for $sets {
            fn from(sets: Vec<$set>) -> Self {
                Self(sets)
            }
        }

        impl From<Vec<Vec<$member>>> for $sets {
            fn from(sets: Vec<Vec<$member>>) -> Self {
                Self(sets.into_iter().map($set::new).collect())
            }
        }

        impl From<(Vec<Vec<$member>>, Vec<usize>)> for $sets {
            fn from((sets, labels): (Vec<Vec<$member>>, Vec<usize>)) -> Self {
                assert_eq!(
                    sets.len(),
                    labels.len(),
                    "There must be exactly one label for each set."
                );
                Self(
                    sets.into_iter()
                        .zip(labels)
                        .map(|(members, label)| $set::new(members).with_label(label))
                        .collect(),
                )
            }
        }
    };
}

sets!(
    NodeSet,
    NodeSets,
    usize,
    nodes,
    "A set of nodes, optionally labeled and named.",
    "Sets of nodes."
);
sets!(
    SideSet,
    SideSets,
    (usize, usize),
    sides,
    "A set of element sides, each an element and the ordinal of one of its faces, optionally labeled and named.",
    "Sets of element sides."
);

impl NodeSet {
    /// The nodes of a mesh whose coordinates satisfy a predicate.
    pub fn select<const D: usize>(
        mesh: &Mesh<D>,
        predicate: impl Fn(&Coordinate<D>) -> bool,
    ) -> Self {
        Self::new(
            mesh.coordinates()
                .iter()
                .enumerate()
                .filter_map(|(node, coordinate)| predicate(coordinate).then_some(node))
                .collect(),
        )
    }
}
