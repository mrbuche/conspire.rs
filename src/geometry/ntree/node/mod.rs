pub mod split;

pub enum Kind<const N: usize, U> {
    Leaf,
    Tree { orthants: [U; N] },
}

pub struct Node<const D: usize, const M: usize, const N: usize, T, U> {
    pub(crate) corner: [T; D],
    pub(crate) length: T,
    pub(crate) facets: [U; M],
    pub(crate) kind: Kind<N, U>,
}

pub type Nodes<const D: usize, const M: usize, const N: usize, T, U> = Vec<Node<D, M, N, T, U>>;
