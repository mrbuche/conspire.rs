//! The surfaces of the union, intersection and differences of two solids.

#[cfg(test)]
mod test;

use super::{Corefined, Region};
use crate::math::FxHashMap;
use crate::{
    geometry::{Coordinate, mesh::Tessellation},
    math::Tensor,
};

/// What to do with a triangle of a surface in a region of the other solid: leave it out, keep
/// it, or keep it turned over.
#[derive(Clone, Copy)]
enum Keep {
    No,
    Yes,
    Flipped,
}

impl Corefined {
    /// The surface of the union of the two solids.
    pub fn union(&self) -> Result<Tessellation, &'static str> {
        self.select(
            |region| match region {
                Region::Outside | Region::Coincident { aligned: true } => Keep::Yes,
                _ => Keep::No,
            },
            |region| match region {
                Region::Outside => Keep::Yes,
                _ => Keep::No,
            },
        )
    }
    /// The surface of the intersection of the two solids.
    pub fn intersection(&self) -> Result<Tessellation, &'static str> {
        self.select(
            |region| match region {
                Region::Inside | Region::Coincident { aligned: true } => Keep::Yes,
                _ => Keep::No,
            },
            |region| match region {
                Region::Inside => Keep::Yes,
                _ => Keep::No,
            },
        )
    }
    /// The surface of the first solid without the second.
    pub fn first_minus_second(&self) -> Result<Tessellation, &'static str> {
        self.select(
            |region| match region {
                Region::Outside | Region::Coincident { aligned: false } => Keep::Yes,
                _ => Keep::No,
            },
            |region| match region {
                Region::Inside => Keep::Flipped,
                _ => Keep::No,
            },
        )
    }
    /// The surface of the second solid without the first.
    pub fn second_minus_first(&self) -> Result<Tessellation, &'static str> {
        self.select(
            |region| match region {
                Region::Inside => Keep::Flipped,
                _ => Keep::No,
            },
            |region| match region {
                Region::Outside | Region::Coincident { aligned: false } => Keep::Yes,
                _ => Keep::No,
            },
        )
    }
    /// The triangles of the two surfaces that are kept, joined at the vertices they share, and
    /// without the vertices that nothing uses.
    ///
    /// Where the surfaces coincide, a triangle is kept from one surface only, so the surfaces of
    /// a union and an intersection do not double up and those of a difference do not leave a
    /// wall.
    fn select(
        &self,
        first: impl Fn(Region) -> Keep,
        second: impl Fn(Region) -> Keep,
    ) -> Result<Tessellation, &'static str> {
        let first_vertices = self.first.mesh().coordinates();
        let second_vertices = self.second.mesh().coordinates();
        let count = first_vertices.iter().count();
        let mut same: FxHashMap<usize, usize> = FxHashMap::default();
        for &[a, b] in &self.shared {
            if same.insert(b, a).is_some_and(|earlier| earlier != a) {
                return Err("a vertex is the same point as two others");
            }
        }
        let second_id = |node: usize| same.get(&node).copied().unwrap_or(count + node);
        let mut faces: Vec<[usize; 3]> = Vec::new();
        let mut gather = |tessellation: &Tessellation,
                          regions: &[Region],
                          keep: &dyn Fn(Region) -> Keep,
                          id: &dyn Fn(usize) -> usize| {
            for (nodes, &region) in tessellation
                .mesh()
                .connectivities()
                .iter()
                .flatten()
                .zip(regions)
            {
                let [a, b, c] = [id(nodes[0]), id(nodes[1]), id(nodes[2])];
                match keep(region) {
                    Keep::No => (),
                    Keep::Yes => faces.push([a, b, c]),
                    Keep::Flipped => faces.push([a, c, b]),
                }
            }
        };
        gather(&self.first, &self.first_regions, &first, &|node| node);
        gather(&self.second, &self.second_regions, &second, &second_id);
        if faces.is_empty() {
            return Err("the surface is empty");
        }
        let positions: Vec<&Coordinate<3>> = first_vertices
            .iter()
            .chain(second_vertices.iter())
            .collect();
        let mut renumbered = FxHashMap::<usize, usize>::default();
        let mut vertices = Vec::<Coordinate<3>>::new();
        let faces: Vec<[usize; 3]> = faces
            .into_iter()
            .map(|face| {
                face.map(|node| {
                    *renumbered.entry(node).or_insert_with(|| {
                        vertices.push(positions[node].clone());
                        vertices.len() - 1
                    })
                })
            })
            .collect();
        Ok((faces, vertices).into())
    }
}
