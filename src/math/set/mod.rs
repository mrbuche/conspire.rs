pub mod dsu;
pub mod sets;
pub mod sets_old;

use crate::math::Tensor;
use std::vec::IntoIter;

pub struct Set<S> {
    members: S,
    numbers: Option<Vec<usize>>,
}

impl<S> Set<S> {
    pub fn members(&self) -> &S {
        &self.members
    }
    pub fn into_members(self) -> S {
        self.members
    }
    pub fn numbers(&self) -> Option<&[usize]> {
        self.numbers.as_deref()
    }
    pub fn set_numbers(&mut self, numbers: Vec<usize>) {
        self.numbers = Some(numbers);
    }
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

impl<S> From<Set<S>> for (S, Option<Vec<usize>>) {
    fn from(set: Set<S>) -> Self {
        (set.members, set.numbers)
    }
}

impl<S, T> IntoIterator for Set<S>
where
    S: IntoIterator<Item = T, IntoIter = IntoIter<T>>,
{
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.members.into_iter()
    }
}

impl<S> Set<S>
where
    S: Tensor,
{
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut S::Item> {
        self.members.iter_mut()
    }
}
