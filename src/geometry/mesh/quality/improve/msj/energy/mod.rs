#[cfg(test)]
mod test;

use super::{
    CAP, Gradient, MARGIN, chunk_size,
    element::{Bound, Frame, Part, bounded},
    kind::KINDS,
    surface::{Plane, Surface},
};
use crate::{
    geometry::{CoordinateList, Coordinates},
    math::{Quantity, QuantityVector, Scalar, Tensor, TensorListVec},
    units::{Area, Dimensionless, Length},
};
use std::{mem::swap, sync::Mutex, thread::scope};

const LIGHT: usize = 1 << 15;

const SAFETY: Scalar = 1.0e-6;

type Active = (usize, CoordinateList<3, 8>);

#[derive(Default)]
pub(super) struct Scratch {
    actives: Vec<Vec<Active>>,
    base: Option<(Coordinates<3>, Quantity)>,
    bounds: Vec<Bound>,
    focus: Option<Vec<usize>>,
    fresh: Vec<Bound>,
    moved: Vec<Quantity<Area>>,
    values: Vec<Quantity<Area>>,
}

pub(super) struct Reach<'a> {
    pub(super) bounds: &'a [Bound],
    pub(super) moved: &'a [Quantity<Area>],
}

impl Reach<'_> {
    pub(super) fn within(&self, index: usize, hex: &[usize; 8]) -> Option<Quantity<Length>> {
        let (margin, shortest) = self.bounds[index];
        let reach = (margin - Dimensionless::of(MARGIN)) * shortest / 12.0 * (1.0 - SAFETY);
        if reach <= Quantity::default() {
            return None;
        }
        let limit = reach * reach;
        let mut farthest = Quantity::default();
        for &node in hex {
            let squared = self.moved[node];
            if squared >= limit {
                return None;
            }
            farthest = farthest.max(squared)
        }
        Some(farthest.sqrt())
    }
    pub(super) fn decayed(&self, index: usize, moved: Quantity<Length>) -> Bound {
        let (margin, shortest) = self.bounds[index];
        let margin = margin - moved / shortest * 12.0;
        if margin > Dimensionless::of(MARGIN) {
            (margin, shortest - moved * 2.0)
        } else {
            Default::default()
        }
    }
}

pub(super) struct Energy<'a> {
    pub(super) boundary: &'a [usize],
    pub(super) hexes: &'a [[usize; 8]],
    pub(super) kinds: &'a [u8],
    pub(super) lambda: TensorListVec<Quantity<Length>, CAP>,
    pub(super) planes: Vec<Plane>,
    pub(super) rho: Quantity,
    pub(super) scales: QuantityVector<Length>,
    pub(super) scratch: Mutex<Scratch>,
    pub(super) theta: Quantity,
}

impl Energy<'_> {
    pub(super) fn refresh(&mut self, surface: &Surface, coordinates: &Coordinates<3>) {
        self.scales = scales(self.hexes, self.kinds, coordinates);
        let sizes = node_scales(self.boundary, self.hexes, &self.scales, coordinates.len());
        self.planes = surface.planes(self.boundary, &sizes, coordinates);
    }
    pub(super) fn typical(&self) -> Quantity<Length> {
        self.scales.iter().copied().sum::<Quantity<Length>>() / self.scales.len().max(1) as Scalar
    }
    pub(super) fn focus(&self, nodes: Option<Vec<usize>>) {
        self.scratch.lock().unwrap().focus = nodes
    }
    pub(super) fn quality(&self, coordinates: &Coordinates<3>) -> Quantity {
        let chunk = chunk_size(self.hexes.len());
        scope(|scope| {
            self.hexes
                .chunks(chunk)
                .zip(self.kinds.chunks(chunk))
                .map(|(hexes, kinds)| {
                    scope.spawn(move || {
                        hexes
                            .iter()
                            .zip(kinds)
                            .map(|(hex, &kind)| {
                                let nodes =
                                    hex.iter().map(|&node| coordinates[node].clone()).collect();
                                let kind = &KINDS[kind as usize];
                                kind.points
                                    .iter()
                                    .map(|point| {
                                        Frame::new(point, &nodes, kind.normalization).scaled
                                    })
                                    .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
                            })
                            .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
        })
    }
    pub(super) fn active(&self, coordinates: &Coordinates<3>, delta: Scalar) -> Vec<bool> {
        self.evaluate(coordinates, Some(&mut Gradient::zero(coordinates.len())));
        let mut active = vec![false; coordinates.len()];
        let scratch = self.scratch.lock().unwrap();
        self.hexes
            .iter()
            .zip(&scratch.bounds)
            .filter(|(_, (margin, _))| margin.value() < delta)
            .for_each(|(hex, _)| hex.iter().for_each(|&node| active[node] = true));
        self.planes
            .iter()
            .for_each(|plane| active[self.boundary[plane.boundary]] = true);
        active
    }
    pub(super) fn gap(&self, coordinates: &Coordinates<3>) -> Quantity {
        let sizes = node_scales(self.boundary, self.hexes, &self.scales, coordinates.len());
        self.planes
            .iter()
            .map(|plane| {
                let node = self.boundary[plane.boundary];
                plane.residual(&coordinates[node]).abs() / sizes[node]
            })
            .fold(Quantity::default(), Quantity::max)
    }
    pub(super) fn settle(&mut self, coordinates: &Coordinates<3>) {
        self.planes.iter().for_each(|plane| {
            let node = self.boundary[plane.boundary];
            self.lambda[plane.boundary][plane.slot] += self.rho * plane.residual(&coordinates[node])
        })
    }
    fn part(
        &self,
        at: usize,
        hex: &[usize; 8],
        coordinates: &Coordinates<3>,
        skip: &Option<Reach>,
        derivative: bool,
    ) -> Part {
        if let Some((skip, moved)) = skip
            .as_ref()
            .and_then(|skip| skip.within(at, hex).map(|moved| (skip, moved)))
        {
            let bound = if derivative {
                skip.decayed(at, moved)
            } else {
                Default::default()
            };
            return (Default::default(), Default::default(), bound);
        }
        let nodes = hex.iter().map(|&node| coordinates[node].clone()).collect();
        let scale = self.scales[at];
        let kind = &KINDS[self.kinds[at] as usize];
        if derivative {
            bounded(kind, self.theta, scale, &nodes, true)
        } else {
            let (value, slopes, _) = bounded(kind, self.theta, scale, &nodes, false);
            (value, slopes, Default::default())
        }
    }
    fn map_values(
        &self,
        coordinates: &Coordinates<3>,
        skip: Option<Reach>,
        values: &mut [Quantity<Area>],
    ) {
        let chunk = chunk_size(self.hexes.len());
        let skip = &skip;
        scope(|scope| {
            values
                .chunks_mut(chunk)
                .zip(self.hexes.chunks(chunk))
                .enumerate()
                .for_each(|(index, (values, hexes))| {
                    scope.spawn(move || {
                        values.iter_mut().zip(hexes).enumerate().for_each(
                            |(offset, (value, hex))| {
                                *value = self
                                    .part(index * chunk + offset, hex, coordinates, skip, false)
                                    .0
                            },
                        )
                    });
                })
        });
    }
    fn map_gradient(
        &self,
        coordinates: &Coordinates<3>,
        skip: Option<Reach>,
        values: &mut [Quantity<Area>],
        bounds: &mut [Bound],
        actives: &mut Vec<Vec<Active>>,
    ) {
        let chunk = chunk_size(self.hexes.len());
        actives.resize_with(self.hexes.len().div_ceil(chunk), Vec::new);
        let skip = &skip;
        scope(|scope| {
            values
                .chunks_mut(chunk)
                .zip(bounds.chunks_mut(chunk))
                .zip(self.hexes.chunks(chunk))
                .zip(actives.iter_mut())
                .enumerate()
                .for_each(|(index, (((values, bounds), hexes), active))| {
                    scope.spawn(move || {
                        active.clear();
                        values
                            .iter_mut()
                            .zip(bounds.iter_mut())
                            .zip(hexes)
                            .enumerate()
                            .for_each(|(offset, ((value, bound), hex))| {
                                let at = index * chunk + offset;
                                let (part, slopes, fresh) =
                                    self.part(at, hex, coordinates, skip, true);
                                *value = part;
                                *bound = fresh;
                                if slopes
                                    .iter()
                                    .any(|slope| (0..3).any(|i| slope[i] != Quantity::default()))
                                {
                                    active.push((at, slopes))
                                }
                            })
                    });
                })
        });
    }
    pub(super) fn evaluate(
        &self,
        coordinates: &Coordinates<3>,
        mut gradient: Option<&mut Gradient>,
    ) -> Quantity<Area> {
        let mut total = Quantity::<Area>::default();
        let mut scratch = self.scratch.lock().unwrap();
        let Scratch {
            actives,
            base,
            bounds,
            focus,
            fresh,
            moved,
            values,
        } = &mut *scratch;
        let skip = match base.as_ref().filter(|(_, theta)| *theta == self.theta) {
            Some((base, _)) => {
                displacements(coordinates, base, moved);
                Some(Reach {
                    bounds,
                    moved: &*moved,
                })
            }
            None => None,
        };
        match gradient.as_deref_mut() {
            Some(gradient) => {
                values.resize_with(self.hexes.len(), Default::default);
                fresh.resize_with(self.hexes.len(), Default::default);
                self.map_gradient(coordinates, skip, values, fresh, actives);
                values.iter().for_each(|&value| total -= value);
                actives.iter().flatten().for_each(|(at, slopes)| {
                    self.hexes[*at]
                        .iter()
                        .zip(slopes.iter())
                        .for_each(|(&node, slope)| gradient[node] -= slope)
                });
                swap(bounds, fresh);
                match (base.as_mut(), focus.as_ref()) {
                    (Some((held, theta)), Some(nodes))
                        if *theta == self.theta && held.len() == coordinates.len() =>
                    {
                        nodes
                            .iter()
                            .for_each(|&node| held[node] = coordinates[node].clone())
                    }
                    _ => *base = Some((coordinates.clone(), self.theta)),
                }
            }
            None => {
                values.resize_with(self.hexes.len(), Default::default);
                self.map_values(coordinates, skip, values);
                values.iter().for_each(|&value| total -= value)
            }
        }
        drop(scratch);
        self.planes.iter().for_each(|plane| {
            let node = self.boundary[plane.boundary];
            let residual = plane.residual(&coordinates[node]);
            let multiplier = self.lambda[plane.boundary][plane.slot];
            total += multiplier * residual + self.rho * residual * residual * 0.5;
            if let Some(gradient) = gradient.as_deref_mut() {
                gradient[node] += &plane.normal * (multiplier + self.rho * residual)
            }
        });
        total
    }
}

pub(super) fn displacements(
    coordinates: &Coordinates<3>,
    base: &Coordinates<3>,
    moved: &mut Vec<Quantity<Area>>,
) {
    moved.resize_with(coordinates.len(), Default::default);
    let chunk = if moved.len() < LIGHT {
        moved.len().max(1)
    } else {
        chunk_size(moved.len())
    };
    scope(|scope| {
        moved
            .chunks_mut(chunk)
            .enumerate()
            .for_each(|(index, moved)| {
                scope.spawn(move || {
                    moved.iter_mut().enumerate().for_each(|(offset, moved)| {
                        let at = index * chunk + offset;
                        let step = &coordinates[at] - &base[at];
                        *moved = &step * &step
                    })
                });
            })
    });
}

pub(super) fn scales(
    hexes: &[[usize; 8]],
    kinds: &[u8],
    coordinates: &Coordinates<3>,
) -> QuantityVector<Length> {
    let chunk = chunk_size(hexes.len());
    scope(|scope| {
        hexes
            .chunks(chunk)
            .zip(kinds.chunks(chunk))
            .map(|(hexes, kinds)| {
                scope.spawn(move || {
                    hexes
                        .iter()
                        .zip(kinds)
                        .map(|(hex, &kind)| {
                            let edges = KINDS[kind as usize].edges;
                            edges
                                .iter()
                                .map(|&[a, b]| (&coordinates[hex[a]] - &coordinates[hex[b]]).norm())
                                .sum::<Quantity<Length>>()
                                / edges.len() as Scalar
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect()
    })
}

pub(super) fn node_scales(
    boundary: &[usize],
    hexes: &[[usize; 8]],
    scales: &QuantityVector<Length>,
    nodes: usize,
) -> QuantityVector<Length> {
    let mut sum = QuantityVector::<Length>::zero(nodes);
    let mut count = vec![0_usize; nodes];
    hexes.iter().zip(scales).for_each(|(hex, &scale)| {
        hex.iter().for_each(|&node| {
            sum[node] += scale;
            count[node] += 1
        })
    });
    let mut sizes = QuantityVector::<Length>::zero(nodes);
    boundary
        .iter()
        .for_each(|&node| sizes[node] = sum[node] / count[node].max(1) as Scalar);
    sizes
}
