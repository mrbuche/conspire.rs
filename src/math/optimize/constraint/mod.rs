pub mod equality;

/// ???
pub trait IntoConstraint<T> {
    fn into_constraint(self, num: usize) -> T;
}
