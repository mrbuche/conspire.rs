use crate::geometry::ntree::{
    Orthotree,
    node::{Kind, Node},
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
    // pub fn leaves_and_facets<'a>(
    //     &self,
    //     node: &'a Node<D, M, N, T, U>,
    // ) -> Option<(&'a [U; N], &'a [Option<U>; M])> {
    //     match &node.kind {
    //         Kind::Leaf => None,
    //         Kind::Tree(orthants) => {
    //             if orthants.iter().any(|&orthant| self[orthant].is_tree()) {
    //                 None
    //             } else {
    //                 Some((orthants, &node.facets))
    //             }
    //         }
    //     }
    // }
    pub fn orthants_leaves(
        &self,
        node: &Node<D, M, N, T, U>,
    ) -> [Option<[Option<U>; N]>; N] {
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
}
