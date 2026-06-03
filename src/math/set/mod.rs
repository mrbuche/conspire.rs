pub mod dsu;
pub mod iter;
pub mod sets;

pub struct Set<S> {
    members: S,
    numbers: Option<Vec<usize>>,
}

impl<S> From<S> for Set<S> {
    fn from(members: S) -> Self {
        Self {
            members,
            numbers: None,
        }
    }
}

impl<S> From<(S, Vec<usize>)> for Set<S> {
    fn from((members, numbers): (S, Vec<usize>)) -> Self {
        Self {
            members,
            numbers: Some(numbers),
        }
    }
}
