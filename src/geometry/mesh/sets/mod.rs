#[cfg(test)]
mod test;

use std::slice::Iter;

macro_rules! collection {
    ($set: ident, $sets: ident, $member: ty, $docs: literal) => {
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
        impl IntoIterator for $sets {
            type Item = $set;
            type IntoIter = std::vec::IntoIter<$set>;
            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
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
            pub fn restrict(&self, map: impl Fn($member) -> Option<$member>) -> Option<Self> {
                let $members: Vec<$member> = self.$members.iter().filter_map(|&m| map(m)).collect();
                (!$members.is_empty()).then(|| Self {
                    label: self.label,
                    name: self.name.clone(),
                    $members,
                })
            }
        }
        collection!($set, $sets, $member, $docs);
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
/// A set of element sides, each an element and the ordinal of one of its faces, optionally labeled
/// and named.
///
/// Once the set is on a mesh it also has the nodes of the face of each side. Sets are equal when
/// they have the same label, name and sides, since the nodes follow from the sides and the mesh.
#[derive(Clone, Debug)]
pub struct SideSet {
    label: Option<usize>,
    name: Option<String>,
    sides: Vec<(usize, usize)>,
    faces: Vec<Vec<usize>>,
}

impl SideSet {
    pub fn new(sides: Vec<(usize, usize)>) -> Self {
        Self {
            label: None,
            name: None,
            sides,
            faces: Vec::new(),
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
    pub fn sides(&self) -> &[(usize, usize)] {
        &self.sides
    }
    /// The nodes of the face of each side, in the order of the sides.
    ///
    /// There are none until the set is on a mesh that has the elements it names.
    pub fn faces(&self) -> &[Vec<usize>] {
        &self.faces
    }
    pub(crate) fn with_faces(mut self, faces: Vec<Vec<usize>>) -> Self {
        assert_eq!(
            faces.len(),
            self.sides.len(),
            "There must be exactly one face for each side."
        );
        self.faces = faces;
        self
    }
    pub fn restrict(&self, map: impl Fn((usize, usize)) -> Option<(usize, usize)>) -> Option<Self> {
        let kept: Vec<(usize, (usize, usize))> = self
            .sides
            .iter()
            .enumerate()
            .filter_map(|(index, &side)| map(side).map(|side| (index, side)))
            .collect();
        (!kept.is_empty()).then(|| Self {
            label: self.label,
            name: self.name.clone(),
            sides: kept.iter().map(|&(_, side)| side).collect(),
            faces: if self.faces.is_empty() {
                Vec::new()
            } else {
                kept.iter()
                    .map(|&(index, _)| self.faces[index].clone())
                    .collect()
            },
        })
    }
}

impl PartialEq for SideSet {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.name == other.name && self.sides == other.sides
    }
}

collection!(SideSet, SideSets, (usize, usize), "Sets of element sides.");
