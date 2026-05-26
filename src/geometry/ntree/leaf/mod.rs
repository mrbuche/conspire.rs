pub mod morton;
pub mod split;

pub enum Kind<const M: usize, const N: usize, U> {
    Leaf,
    Tree {
        facets: [U; M],
        orthants: [U; N],
    },
}

pub struct Node<const D: usize, const M: usize, const N: usize, T, U> {
    pub(crate) corner: [T; D],
    pub(crate) length: T,
    pub(crate) kind: Kind<M, N, U>,
}
