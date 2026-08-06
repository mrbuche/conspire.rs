#[cfg(test)]
mod test;

use super::{Class, DIRECTIONS, PADDING};
use crate::{
    geometry::{
        Coordinate, CoordinateList, CoordinatesRef,
        bbox::BoundingBox,
        mesh::{
            Mesh,
            tessellation::{D, Tessellation},
        },
    },
    math::{FxHashMap, FxHashSet, Scalar, Tensor},
};
use std::array::from_fn;

const NEIGHBORS: [[isize; D]; 6] = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
];

pub(super) struct Lattice {
    cells: FxHashMap<[usize; D], Class>,
    nel: [usize; D],
    origin: Coordinate<D>,
    spacing: Scalar,
}

impl Lattice {
    fn cell(&self, [i, j, k]: [usize; D]) -> BoundingBox<D> {
        let index = [i, j, k];
        let minimum = Coordinate::const_from(from_fn(|d| {
            self.origin[d] + index[d] as Scalar * self.spacing
        }));
        let maximum = Coordinate::const_from(from_fn(|d| minimum[d] + self.spacing));
        BoundingBox::from(CoordinateList::from([minimum, maximum]))
    }
    fn centroid(&self, index: [usize; D]) -> Coordinate<D> {
        Coordinate::const_from(from_fn(|d| {
            self.origin[d] + (index[d] as Scalar + 0.5) * self.spacing
        }))
    }
    fn neighbors(&self, index: [usize; D]) -> impl Iterator<Item = [usize; D]> + '_ {
        NEIGHBORS.iter().filter_map(move |offset| {
            let mut next = [0; D];
            (0..D)
                .all(|d| {
                    let moved = index[d] as isize + offset[d];
                    next[d] = moved as usize;
                    moved >= 0 && moved < self.nel[d] as isize
                })
                .then_some(next)
        })
    }
    pub(super) fn cells(&self) -> Vec<([usize; D], Class)> {
        let mut cells: Vec<_> = self
            .cells
            .iter()
            .map(|(&index, &class)| (index, class))
            .collect();
        cells.sort_unstable_by_key(|&([i, j, k], _)| (k, j, i));
        cells
    }
    pub(super) fn frame(&self) -> (Coordinate<D>, Scalar) {
        (self.origin.clone(), self.spacing)
    }
    pub(super) fn mesh(&self) -> Mesh<D> {
        Mesh::from_lattice_cells(
            self.cells().into_iter().map(|(index, _)| (index, 1)),
            self.nel,
            &Coordinate::const_from([self.spacing; D]),
            &self.origin,
        )
    }
}

impl Tessellation {
    pub(super) fn lattice(&self, spacing: Scalar) -> Result<Lattice, &'static str> {
        self.lattice_shifted(spacing, [0.0; D])
    }
    pub(super) fn lattice_shifted(
        &self,
        spacing: Scalar,
        shift: [Scalar; D],
    ) -> Result<Lattice, &'static str> {
        if spacing <= 0.0 || spacing.is_nan() {
            return Err("lattice spacing must be positive");
        }
        let surface = self.mesh();
        let coordinates = surface.coordinates();
        let bounds = BoundingBox::from(coordinates.clone());
        let origin = Coordinate::const_from(from_fn(|d| {
            bounds.minimum()[d] - (PADDING as Scalar + shift[d]) * spacing
        }));
        let nel = from_fn(|d| {
            ((bounds.maximum()[d] - bounds.minimum()[d]) / spacing).ceil() as usize
                + 2 * PADDING as usize
        });
        let mut lattice = Lattice {
            cells: FxHashMap::default(),
            nel,
            origin,
            spacing,
        };
        lattice.rasterize(surface);
        lattice.fill(self, surface)?;
        lattice.enclose();
        Ok(lattice)
    }
}

impl Lattice {
    fn rasterize(&mut self, surface: &Mesh<D>) {
        let coordinates = surface.coordinates();
        surface
            .connectivities()
            .iter()
            .flatten()
            .for_each(|triangle| {
                let corners: [&Coordinate<D>; 3] = from_fn(|corner| &coordinates[triangle[corner]]);
                let low: [usize; D] = from_fn(|d| {
                    let minimum = corners.iter().fold(Scalar::INFINITY, |a, c| a.min(c[d]));
                    (((minimum - self.origin[d]) / self.spacing).floor() as isize - 1)
                        .clamp(0, self.nel[d] as isize - 1) as usize
                });
                let high: [usize; D] = from_fn(|d| {
                    let maximum = corners
                        .iter()
                        .fold(Scalar::NEG_INFINITY, |a, c| a.max(c[d]));
                    (((maximum - self.origin[d]) / self.spacing).floor() as isize + 1)
                        .clamp(0, self.nel[d] as isize - 1) as usize
                });
                for k in low[2]..=high[2] {
                    for j in low[1]..=high[1] {
                        for i in low[0]..=high[0] {
                            if self
                                .cell([i, j, k])
                                .overlaps_triangle(corners[0], corners[1], corners[2])
                            {
                                self.cells.insert([i, j, k], Class::Cut);
                            }
                        }
                    }
                }
            });
    }
    fn fill(&mut self, tessellation: &Tessellation, surface: &Mesh<D>) -> Result<(), &'static str> {
        if self.cells.is_empty() {
            return Err("surface does not intersect the lattice");
        }
        let coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: CoordinatesRef<'_, D> = tessellation.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let mut seeds: Vec<[usize; D]> = self
            .cells
            .keys()
            .flat_map(|&index| self.neighbors(index))
            .filter(|index| !self.cells.contains_key(index))
            .collect();
        seeds.sort_unstable_by_key(|&[i, j, k]| (k, j, i));
        seeds.dedup();
        let mut exterior = FxHashSet::default();
        let mut stack = Vec::new();
        for seed in seeds {
            if self.cells.contains_key(&seed) || exterior.contains(&seed) {
                continue;
            }
            if !tessellation.encloses(
                &self.centroid(seed),
                coordinates,
                &elements,
                &normals,
                &directions,
            ) {
                exterior.insert(seed);
                continue;
            }
            self.cells.insert(seed, Class::Inside);
            stack.push(seed);
            while let Some(index) = stack.pop() {
                let next: Vec<_> = self
                    .neighbors(index)
                    .filter(|next| !self.cells.contains_key(next))
                    .collect();
                next.into_iter().for_each(|next| {
                    self.cells.insert(next, Class::Inside);
                    stack.push(next);
                });
            }
        }
        Ok(())
    }
    fn enclose(&mut self) {
        let outside: Vec<[usize; D]> = self
            .cells
            .keys()
            .flat_map(|&index| self.neighbors(index))
            .filter(|index| !self.cells.contains_key(index))
            .collect();
        outside.into_iter().for_each(|index| {
            self.cells.insert(index, Class::Outside);
        });
    }
}
