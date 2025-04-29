pub mod equality;

/// ???
pub trait ToConstraint<T> {
    fn to_constraint(&self, num: usize) -> T;
}
