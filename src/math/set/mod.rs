#[cfg(test)]
mod test;

pub mod dsu;
pub mod sets;

use std::{array, slice, vec};

#[derive(Clone)]
pub struct Set<S, T>
where
    S: IntoIterator<Item = T>,
{
    members: S,
}

impl<S, T> From<S> for Set<S, T>
where
    S: IntoIterator<Item = T>,
{
    fn from(members: S) -> Self {
        Self { members }
    }
}

impl<T> From<Set<Vec<T>, T>> for Vec<T> {
    fn from(set: Set<Vec<T>, T>) -> Self {
        set.members
    }
}

impl<const N: usize, T> IntoIterator for Set<[T; N], T> {
    type Item = T;
    type IntoIter = array::IntoIter<Self::Item, N>;
    fn into_iter(self) -> Self::IntoIter {
        self.members.into_iter()
    }
}

impl<'a, const N: usize, T> IntoIterator for &'a Set<[T; N], T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.members.iter()
    }
}

impl<T> IntoIterator for Set<Vec<T>, T> {
    type Item = T;
    type IntoIter = vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.members.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Set<Vec<T>, T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.members.iter()
    }
}
