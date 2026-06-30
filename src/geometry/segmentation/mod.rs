#[cfg(test)]
mod test;

use crate::geometry::{Coordinate, grid::Grid};
use std::{
    array::from_fn,
    ops::{Deref, DerefMut, Range},
};

pub type Segmentation2D<T> = Segmentation<2, T>;
pub type Segmentation3D<T> = Segmentation<3, T>;

pub struct Segmentation<const D: usize, T> {
    grid: Grid<D, T>,
    scale: Coordinate<D>,
    translate: Coordinate<D>,
}

impl<const D: usize, T> Segmentation<D, T> {
    pub fn new(grid: Grid<D, T>, scale: Coordinate<D>, translate: Coordinate<D>) -> Self {
        assert!(
            (0..D).all(|axis| scale[axis] > 0.0),
            "scale must be positive in every direction"
        );
        Self {
            grid,
            scale,
            translate,
        }
    }
    pub fn grid(&self) -> &Grid<D, T> {
        &self.grid
    }
    pub fn into_parts(self) -> (Grid<D, T>, Coordinate<D>, Coordinate<D>) {
        (self.grid, self.scale, self.translate)
    }
    pub fn scale(&self) -> &Coordinate<D> {
        &self.scale
    }
    pub fn translate(&self) -> &Coordinate<D> {
        &self.translate
    }
}

impl<const D: usize, T: Copy> Segmentation<D, T> {
    pub fn extract(&self, ranges: [Range<usize>; D]) -> Self {
        let translate = from_fn::<_, D, _>(|axis| {
            self.translate[axis] + ranges[axis].start as f64 * self.scale[axis]
        })
        .into();
        Self {
            grid: self.grid.extract(ranges),
            scale: self.scale.clone(),
            translate,
        }
    }
}

impl<const D: usize, T> From<Grid<D, T>> for Segmentation<D, T> {
    fn from(grid: Grid<D, T>) -> Self {
        Self {
            grid,
            scale: [1.0; D].into(),
            translate: [0.0; D].into(),
        }
    }
}

impl<const D: usize, T> Deref for Segmentation<D, T> {
    type Target = Grid<D, T>;
    fn deref(&self) -> &Self::Target {
        &self.grid
    }
}

impl<const D: usize, T> DerefMut for Segmentation<D, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.grid
    }
}
