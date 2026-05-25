pub mod morton;
pub mod split;

#[derive(Copy, Clone)]
pub struct Leaf<const D: usize, T, U> {
    pub(crate) corner: [T; D],
    pub(crate) length: T,
    pub(crate) data: U,
}

pub type Leaves<const D: usize, T, U> = Vec<Leaf<D, T, U>>;
