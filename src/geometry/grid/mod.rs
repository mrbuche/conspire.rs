#[cfg(test)]
mod test;

mod read;
mod write;

pub use self::{read::Input, write::Output};

use std::ops::{Index, IndexMut};

pub type Pixels<T> = Grid<2, T>;
pub type Voxels<T> = Grid<3, T>;

pub struct Grid<const D: usize, T> {
    data: Vec<T>,
    nel: [usize; D],
}

impl<const D: usize, T> Grid<D, T> {
    pub fn new(data: Vec<T>, nel: [usize; D]) -> Self {
        assert_eq!(
            data.len(),
            nel.iter().product::<usize>(),
            "voxel data length must equal the product of nel"
        );
        Self { data, nel }
    }
    pub fn data(&self) -> &[T] {
        &self.data
    }
    pub fn flat(&self, index: [usize; D]) -> usize {
        let mut offset = 0;
        let mut stride = 1;
        for (&i, &n) in index.iter().zip(&self.nel) {
            offset += i * stride;
            stride *= n;
        }
        offset
    }
    pub fn nel(&self) -> &[usize; D] {
        &self.nel
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<const D: usize, T> Index<[usize; D]> for Grid<D, T> {
    type Output = T;
    fn index(&self, index: [usize; D]) -> &T {
        &self.data[self.flat(index)]
    }
}

impl<const D: usize, T> IndexMut<[usize; D]> for Grid<D, T> {
    fn index_mut(&mut self, index: [usize; D]) -> &mut T {
        let offset = self.flat(index);
        &mut self.data[offset]
    }
}
