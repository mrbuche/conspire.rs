#[cfg(test)]
mod test;

use super::{Mesh, Tessellation};
use crate::{
    geometry::Coordinates,
    math::{Scalar, Tensor},
};
use std::collections::{HashMap, HashSet};

const TOLERANCE: Scalar = 1.0e-6;

pub(crate) struct Symmetry {
    center: [Scalar; 3],
    images: Vec<Vec<usize>>,
    signs: Vec<[Scalar; 3]>,
}

pub(super) struct Orbits {
    center: [Scalar; 3],
    images: Vec<Vec<usize>>,
    signs: Vec<[Scalar; 3]>,
}

struct Lookup<'a> {
    cell: Scalar,
    cells: HashMap<[i64; 3], Vec<usize>>,
    points: &'a [[Scalar; 3]],
    tolerance: Scalar,
}

impl<'a> Lookup<'a> {
    fn new(points: &'a [[Scalar; 3]], tolerance: Scalar) -> Self {
        let cell = 2.0 * tolerance;
        let mut cells = HashMap::<[i64; 3], Vec<usize>>::new();
        points
            .iter()
            .enumerate()
            .for_each(|(index, point)| cells.entry(key(point, cell)).or_default().push(index));
        Self {
            cell,
            cells,
            points,
            tolerance,
        }
    }
    fn find(&self, point: &[Scalar; 3]) -> Option<usize> {
        let [i, j, k] = key(point, self.cell);
        (-1..=1)
            .flat_map(|di| {
                (-1..=1).flat_map(move |dj| (-1..=1).map(move |dk| [i + di, j + dj, k + dk]))
            })
            .filter_map(|neighbor| self.cells.get(&neighbor))
            .flatten()
            .map(|&index| (index, distance(&self.points[index], point)))
            .filter(|&(_, distance)| distance <= self.tolerance)
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
    }
}

fn key(point: &[Scalar; 3], cell: Scalar) -> [i64; 3] {
    point.map(|entry| (entry / cell).floor() as i64)
}

fn distance(a: &[Scalar; 3], b: &[Scalar; 3]) -> Scalar {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<Scalar>().sqrt()
}

fn points(coordinates: &Coordinates<3>) -> Vec<[Scalar; 3]> {
    coordinates
        .iter()
        .map(|point| [point[0].value(), point[1].value(), point[2].value()])
        .collect()
}

fn reflect(point: &[Scalar; 3], center: &[Scalar; 3], axis: usize) -> [Scalar; 3] {
    let mut reflected = *point;
    reflected[axis] = 2.0 * center[axis] - point[axis];
    reflected
}

fn mirror(
    points: &[[Scalar; 3]],
    lookup: &Lookup,
    center: &[Scalar; 3],
    axis: usize,
) -> Option<Vec<usize>> {
    let images: Vec<usize> = points
        .iter()
        .map(|point| lookup.find(&reflect(point, center, axis)))
        .collect::<Option<_>>()?;
    let distinct: HashSet<usize> = images.iter().copied().collect();
    (distinct.len() == points.len()).then_some(images)
}

fn elements(mesh: &Mesh<3>) -> Vec<HashSet<Vec<usize>>> {
    mesh.iter()
        .map(|block| {
            block
                .iter()
                .map(|element| {
                    let mut nodes = element.to_vec();
                    nodes.sort_unstable();
                    nodes
                })
                .collect()
        })
        .collect()
}

fn invariant(mesh: &Mesh<3>, images: &[usize]) -> bool {
    mesh.iter().zip(elements(mesh)).all(|(block, expected)| {
        block.iter().all(|element| {
            let mut nodes: Vec<usize> = element.iter().map(|&node| images[node]).collect();
            nodes.sort_unstable();
            expected.contains(&nodes)
        })
    })
}

impl Symmetry {
    pub(crate) fn detect(mesh: &Mesh<3>, target: &Tessellation) -> Option<Self> {
        let nodes = points(mesh.coordinates());
        let surface = target.mesh();
        let vertices = points(surface.coordinates());
        let (low, high) = nodes.iter().fold(
            ([Scalar::INFINITY; 3], [Scalar::NEG_INFINITY; 3]),
            |(low, high), point| {
                (
                    [0, 1, 2].map(|i| low[i].min(point[i])),
                    [0, 1, 2].map(|i| high[i].max(point[i])),
                )
            },
        );
        let center = [0, 1, 2].map(|i| 0.5 * (low[i] + high[i]));
        let tolerance = TOLERANCE * (0..3).map(|i| high[i] - low[i]).fold(0.0, Scalar::max);
        if nodes.is_empty() || tolerance <= 0.0 {
            return None;
        }
        let node_lookup = Lookup::new(&nodes, tolerance);
        let mirrors: Vec<(usize, Vec<usize>)> = (0..3)
            .filter_map(|axis| {
                mirror(&nodes, &node_lookup, &center, axis)
                    .filter(|images| invariant(mesh, images))
                    .map(|images| (axis, images))
            })
            .collect();
        if mirrors.is_empty() {
            return None;
        }
        let vertex_lookup = Lookup::new(&vertices, tolerance);
        let mirrors: Vec<(usize, Vec<usize>)> = mirrors
            .into_iter()
            .filter(|&(axis, _)| {
                mirror(&vertices, &vertex_lookup, &center, axis)
                    .is_some_and(|images| invariant(surface, &images))
            })
            .collect();
        if mirrors.is_empty() {
            return None;
        }
        let count = nodes.len();
        let (images, signs) = (0..1_usize << mirrors.len())
            .map(|subset| {
                let mut images: Vec<usize> = (0..count).collect();
                let mut signs = [1.0; 3];
                mirrors
                    .iter()
                    .enumerate()
                    .filter(|&(bit, _)| subset >> bit & 1 == 1)
                    .for_each(|(_, (axis, mirrored))| {
                        images
                            .iter_mut()
                            .for_each(|image| *image = mirrored[*image]);
                        signs[*axis] = -1.0;
                    });
                (images, signs)
            })
            .unzip();
        Some(Self {
            center,
            images,
            signs,
        })
    }
    pub(crate) fn extend(
        mut self,
        duplicates: &HashMap<usize, usize>,
        count: usize,
    ) -> Option<Self> {
        for images in self.images.iter_mut() {
            let original = images.len();
            images.resize(count, 0);
            for (&node, &duplicate) in duplicates {
                images[duplicate] = *duplicates.get(&images[node])?;
            }
            if (original..count).any(|node| images[node] < original) {
                return None;
            }
        }
        Some(self)
    }
    pub(super) fn orbits(&self, nodes: &[usize]) -> Option<Orbits> {
        let mut slot = vec![None; self.images.first().map_or(0, Vec::len)];
        nodes
            .iter()
            .enumerate()
            .for_each(|(position, &node)| slot[node] = Some(position));
        let images = self
            .images
            .iter()
            .map(|images| {
                nodes
                    .iter()
                    .map(|&node| slot[images[node]])
                    .collect::<Option<Vec<usize>>>()
            })
            .collect::<Option<_>>()?;
        Some(Orbits {
            center: self.center,
            images,
            signs: self.signs.clone(),
        })
    }
}

impl Orbits {
    pub(super) fn points(&self, field: &[[Scalar; 3]]) -> Vec<[Scalar; 3]> {
        self.average(field, true)
    }
    pub(super) fn vectors(&self, field: &[[Scalar; 3]]) -> Vec<[Scalar; 3]> {
        self.average(field, false)
    }
    fn average(&self, field: &[[Scalar; 3]], affine: bool) -> Vec<[Scalar; 3]> {
        let scale = 1.0 / self.signs.len() as Scalar;
        (0..field.len())
            .map(|position| {
                let mut sum = [0.0; 3];
                self.images
                    .iter()
                    .zip(&self.signs)
                    .for_each(|(images, signs)| {
                        let entry = &field[images[position]];
                        (0..3).for_each(|i| {
                            sum[i] += if affine {
                                self.center[i] + signs[i] * (entry[i] - self.center[i])
                            } else {
                                signs[i] * entry[i]
                            }
                        })
                    });
                sum.map(|entry| entry * scale)
            })
            .collect()
    }
}
