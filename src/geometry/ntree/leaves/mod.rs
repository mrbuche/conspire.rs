use crate::geometry::ntree::{
    Orthotree,
    node::{Kind, Node},
    subdivide::insert_bit,
};
use std::array::from_fn;

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U>
    Orthotree<D, L, M, N, T, U>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    pub fn all_leaves<'a>(&self, node: &'a Node<D, M, N, T, U>) -> Option<&'a [U; N]> {
        match &node.kind {
            Kind::Leaf => None,
            Kind::Tree(orthants) => {
                if orthants.iter().any(|&orthant| self[orthant].is_tree()) {
                    None
                } else {
                    Some(orthants)
                }
            }
        }
    }
    pub fn leaves(&self, node: &Node<D, M, N, T, U>) -> [Option<U>; N] {
        match &node.kind {
            Kind::Leaf => [None; N],
            Kind::Tree(orthants) => from_fn(|i| {
                if self[orthants[i]].is_leaf() {
                    Some(orthants[i])
                } else {
                    None
                }
            }),
        }
    }
    pub fn leaves_on_facet(&self, node: &Node<D, M, N, T, U>, facet: usize) -> [Option<U>; L] {
        let (axis, side) = (facet >> 1, facet & 1);
        match &node.kind {
            Kind::Leaf => from_fn(|_| None),
            Kind::Tree(orthants) => from_fn(|i| {
                let orthant = orthants[insert_bit(i, axis, side)];
                self[orthant].is_leaf().then_some(orthant)
            }),
        }
    }
    pub fn leaves_and_facets(
        &self,
        node: &Node<D, M, N, T, U>,
    ) -> [Option<(U, [Option<U>; D])>; N] {
        match &node.kind {
            Kind::Leaf => from_fn(|_| None),
            Kind::Tree(orthants) => from_fn(|i| {
                let orthant = orthants[i];
                if self[orthant].is_leaf() {
                    let facets = &self[orthant].facets;
                    let external: [Option<U>; D] = from_fn(|b| facets[2 * b + ((i >> b) & 1)]);
                    Some((orthant, external))
                } else {
                    None
                }
            }),
        }
    }
    pub fn orthants_leaves(&self, node: &Node<D, M, N, T, U>) -> [Option<[Option<U>; N]>; N] {
        match &node.kind {
            Kind::Leaf => from_fn(|_| None),
            Kind::Tree(orthants) => from_fn(|i| match &self[orthants[i]].kind {
                Kind::Leaf => None,
                Kind::Tree(sub_orthants) => {
                    let inner: [Option<U>; N] = from_fn(|j| {
                        if self[sub_orthants[j]].is_leaf() {
                            Some(sub_orthants[j])
                        } else {
                            None
                        }
                    });
                    if inner.iter().any(|x| x.is_some()) {
                        Some(inner)
                    } else {
                        None
                    }
                }
            }),
        }
    }
    pub fn orthants_leaves_on_facet(
        &self,
        node: &Node<D, M, N, T, U>,
        face: usize,
    ) -> [Option<[Option<U>; L]>; L] {
        let (axis, side) = (face >> 1, face & 1);
        match &node.kind {
            Kind::Leaf => from_fn(|_| None),
            Kind::Tree(orthants) => {
                from_fn(|i| match &self[orthants[insert_bit(i, axis, side)]].kind {
                    Kind::Leaf => None,
                    Kind::Tree(sub_orthants) => {
                        let inner: [Option<U>; L] = from_fn(|j| {
                            let leaf = sub_orthants[insert_bit(j, axis, side)];
                            self[leaf].is_leaf().then_some(leaf)
                        });
                        inner.iter().any(|x| x.is_some()).then_some(inner)
                    }
                })
            }
        }
    }
    pub fn orthants_all_leaves_on_facet(
        &self,
        node: &Node<D, M, N, T, U>,
        face: usize,
    ) -> Option<[[U; L]; L]> {
        let orthants_leaves = self.orthants_leaves_on_facet(node, face);
        orthants_leaves
            .iter()
            .all(|&orthant| orthant.is_some_and(|leaves| leaves.iter().all(Option::is_some)))
            .then(|| from_fn(|i| from_fn(|j| orthants_leaves[i].unwrap()[j].unwrap())))
    }
}
