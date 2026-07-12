use crate::math::set::Set;
use std::cell::OnceCell;

pub struct Sets<S> {
    converse: OnceCell<Vec<Vec<usize>>>,
    set: Set<S>,
}

impl<S> Sets<S> {
    pub fn converse<T>(&self) -> &[Vec<usize>]
    where
        S: AsRef<[T]>,
        T: AsRef<[usize]>,
    {
        self.converse.get_or_init(|| {
            let num_inner = self
                .members()
                .as_ref()
                .iter()
                .flat_map(|row| row.as_ref().iter().copied())
                .max()
                .map_or(0, |m| m + 1);
            let mut converse = vec![Vec::new(); num_inner];
            for (outer, row) in self.members().as_ref().iter().enumerate() {
                for &inner in row.as_ref() {
                    converse[inner].push(outer);
                }
            }
            converse
        })
    }
    pub fn members(&self) -> &S {
        self.set.members()
    }
    pub fn into_members(self) -> S {
        self.set.into_members()
    }
    pub fn numbers(&self) -> Option<&[usize]> {
        self.set.numbers()
    }
    pub fn set(&self) -> &Set<S> {
        &self.set
    }
    pub fn set_numbers(&mut self, numbers: Vec<usize>) {
        self.set.set_numbers(numbers);
    }
}

impl<S> From<S> for Sets<S> {
    fn from(members: S) -> Self {
        Set::from(members).into()
    }
}

impl<S> From<Set<S>> for Sets<S> {
    fn from(set: Set<S>) -> Self {
        Self {
            converse: OnceCell::new(),
            set,
        }
    }
}

impl<S> From<Sets<S>> for Set<S> {
    fn from(sets: Sets<S>) -> Self {
        sets.set
    }
}
