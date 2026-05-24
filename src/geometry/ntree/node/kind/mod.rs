use crate::geometry::ntree::node::orthants::Orthants;

pub enum Kind<const N: usize, U> {
    Leaf,
    Tree(Orthants<N, U>),
}

impl<const N: usize, U> From<[U; N]> for Kind<N, U> {
    fn from(array: [U; N]) -> Self {
        Kind::Tree(array.into())
    }
}
