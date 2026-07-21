#[cfg(test)]
mod test;

mod defeature;
mod from;
mod read;
mod write;

pub use self::{read::Input, write::Output};

use std::{
    array,
    borrow::Cow,
    iter,
    ops::{Index, IndexMut, Range},
};

pub type Pixels<T> = Grid<2, T>;
pub type Voxels<T> = Grid<3, T>;

pub struct Grid<const D: usize, T> {
    data: Vec<T>,
    nel: [usize; D],
    strides: [usize; D],
}

fn col_major_strides<const D: usize>(nel: [usize; D]) -> [usize; D] {
    let mut strides = [1usize; D];
    for axis in 1..D {
        strides[axis] = strides[axis - 1] * nel[axis - 1];
    }
    strides
}

fn row_major_strides<const D: usize>(nel: [usize; D]) -> [usize; D] {
    let mut strides = [1usize; D];
    for axis in (0..D.saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * nel[axis + 1];
    }
    strides
}

impl<const D: usize, T> Grid<D, T> {
    pub fn new(data: Vec<T>, nel: [usize; D]) -> Self {
        assert_eq!(
            data.len(),
            nel.iter().product::<usize>(),
            "voxel data length must equal the product of nel"
        );
        Self {
            data,
            nel,
            strides: col_major_strides(nel),
        }
    }
    pub fn new_row_major(data: Vec<T>, nel: [usize; D]) -> Self {
        assert_eq!(
            data.len(),
            nel.iter().product::<usize>(),
            "voxel data length must equal the product of nel"
        );
        Self {
            data,
            nel,
            strides: row_major_strides(nel),
        }
    }
    pub fn data(&self) -> &[T] {
        &self.data
    }
    pub fn flat(&self, index: [usize; D]) -> usize {
        index
            .iter()
            .zip(&self.strides)
            .map(|(&i, &stride)| i * stride)
            .sum()
    }
    fn axes_by_stride(&self) -> [usize; D] {
        let mut order: [usize; D] = array::from_fn(|axis| axis);
        order.sort_by_key(|&axis| self.strides[axis]);
        order
    }
    pub fn logical_iter<'a>(&'a self) -> impl Iterator<Item = ([usize; D], &'a T)> + 'a {
        let nel = self.nel;
        let order = self.axes_by_stride();
        let mut index = [0usize; D];
        let mut offset = 0usize;
        iter::from_fn(move || {
            if offset == self.data.len() {
                return None;
            }
            let item = (index, &self.data[offset]);
            offset += 1;
            for &axis in &order {
                index[axis] += 1;
                if index[axis] < nel[axis] {
                    break;
                }
                index[axis] = 0;
            }
            Some(item)
        })
    }
    pub fn nel(&self) -> &[usize; D] {
        &self.nel
    }
    pub fn is_col_major(&self) -> bool {
        self.strides == col_major_strides(self.nel)
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<const D: usize, T: Copy> Grid<D, T> {
    pub fn data_col_major(&self) -> Cow<'_, [T]> {
        if self.strides == col_major_strides(self.nel) {
            Cow::Borrowed(&self.data)
        } else {
            Cow::Owned(read::transpose(self.data.clone(), self.nel))
        }
    }
}

impl<const D: usize, T: Copy> Grid<D, T> {
    pub fn extract(&self, ranges: [Range<usize>; D]) -> Self {
        let nel: [usize; D] = array::from_fn(|axis| ranges[axis].len());
        let total: usize = nel.iter().product();
        let mut data = Vec::with_capacity(total);
        let mut index = [0usize; D];
        for offset in 0..total {
            let mut remainder = offset;
            for axis in 0..D {
                index[axis] = ranges[axis].start + remainder % nel[axis];
                remainder /= nel[axis];
            }
            data.push(self[index]);
        }
        Self::new(data, nel)
    }
}

impl<const D: usize, T: PartialEq> Grid<D, T> {
    pub fn diff(&self, other: &Self) -> Grid<D, u8> {
        assert_eq!(
            self.nel(),
            other.nel(),
            "grids do not have the same dimensions"
        );
        if self.strides == other.strides {
            let data = self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| (a != b) as u8)
                .collect();
            Grid {
                data,
                nel: self.nel,
                strides: self.strides,
            }
        } else {
            let nel = self.nel;
            let data = (0..self.data.len())
                .map(|offset| {
                    let mut index = [0usize; D];
                    let mut remainder = offset;
                    for axis in 0..D {
                        index[axis] = remainder % nel[axis];
                        remainder /= nel[axis];
                    }
                    (self[index] != other[index]) as u8
                })
                .collect();
            Grid::new(data, nel)
        }
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
