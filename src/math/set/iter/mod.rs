#[cfg(test)]
mod test;

use crate::math::set::Set;
use std::{
    iter::{Enumerate, Zip},
    vec::IntoIter,
};

impl<S> Set<S> {
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (usize, <&'a S as IntoIterator>::Item)> + 'a
    where
        &'a S: IntoIterator,
    {
        match &self.numbers {
            None => SetIter::Enumerated((&self.members).into_iter().enumerate()),
            Some(numbers) => SetIter::Numbered(numbers.iter().copied().zip(&self.members)),
        }
    }
}

impl<S> IntoIterator for Set<S>
where
    S: IntoIterator,
{
    type Item = (usize, S::Item);
    type IntoIter = SetIter<Enumerate<S::IntoIter>, Zip<IntoIter<usize>, S::IntoIter>>;
    fn into_iter(self) -> Self::IntoIter {
        match self.numbers {
            None => SetIter::Enumerated(self.members.into_iter().enumerate()),
            Some(numbers) => SetIter::Numbered(numbers.into_iter().zip(self.members)),
        }
    }
}

pub enum SetIter<E, N> {
    Enumerated(E),
    Numbered(N),
}

impl<E, N, T> Iterator for SetIter<E, N>
where
    E: Iterator<Item = T>,
    N: Iterator<Item = T>,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        match self {
            Self::Enumerated(iter) => iter.next(),
            Self::Numbered(iter) => iter.next(),
        }
    }
}
