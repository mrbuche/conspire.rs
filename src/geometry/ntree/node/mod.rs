pub mod orthants;
pub mod split;
pub mod subdivide;

pub enum Kind<const N: usize, U> {
    Leaf,
    Tree(Orthants<N, U>),
}

pub struct Node<const D: usize, const M: usize, const N: usize, T, U, V = ()> {
    pub(crate) corner: [T; D],
    pub(crate) length: T,
    pub(crate) facets: [Option<U>; M],
    pub(crate) kind: Kind<N, U>,
    pub(crate) value: Option<V>,
}

pub type Nodes<const D: usize, const M: usize, const N: usize, T, U, V = ()> =
    Vec<Node<D, M, N, T, U, V>>;

pub type Orthants<const N: usize, U> = [U; N];
