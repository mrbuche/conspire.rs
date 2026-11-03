#[cfg(test)]
mod test;

mod element;
mod kind;

use crate::{
    geometry::{
        Coordinate, CoordinateList, Coordinates, CoordinatesRef, Direction, DirectionsRef,
        bbox::BoundingBox,
        mesh::{Connectivity, Mesh, MeshError, Tessellation},
    },
    math::{
        Quantity, QuantityVector, Reference, Scalar, Tensor, TensorListVec, TensorRank1Vec,
        optimize::{Armijo, Corrections, LimitedMemoryBFGS, Tolerances},
    },
    units::{Area, Dimensionless, Length},
};
use element::{Bound, Frame, Part, bounded};
use kind::{HEXAHEDRON, KINDS, PYRAMID, TETRAHEDRON, WEDGE};
use std::{
    array::from_fn,
    cell::RefCell,
    mem::swap,
    sync::Mutex,
    thread::{available_parallelism, scope},
};

type Gradient = TensorRank1Vec<3, Reference, Length>;

const CAP: usize = 4;
const DELTA: Scalar = 0.05;
const LIGHT: usize = 1 << 15;
const SERIAL: usize = 512;
const CREASE: Quantity = Dimensionless::of(0.866);
const FIRST: Quantity = Dimensionless::of(0.01);
const RADIUS: Quantity = Dimensionless::of(0.2);
const MARGIN: Scalar = 1.0e-9;
const SAFETY: Scalar = 1.0e-6;
const RHO: Quantity = Dimensionless::of(1.0e2);
const STALL: Quantity = Dimensionless::of(1.0e-6);
const STEP: Quantity = Dimensionless::of(0.01);
const TIE: Scalar = 1.0e-6;
const TOLERANCE: Quantity = Dimensionless::of(1.0e-3);

fn solver(typical: Quantity<Length>) -> LimitedMemoryBFGS<Armijo, Length> {
    LimitedMemoryBFGS {
        abs_tol: Tolerances {
            constraint: 0.0,
            residual: 1.0e-14,
        },
        history: 10,
        line_search: Armijo {
            control: 1.0e-4,
            cut_back: 0.5,
            max_steps: 27,
        },
        max_steps: 50,
        rel_tol: None,
        typical,
    }
}

struct Plane {
    boundary: usize,
    normal: Direction<3>,
    offset: Quantity<Length>,
    slot: usize,
}

type Active = (usize, CoordinateList<3, 8>);
#[derive(Default)]
struct Scratch {
    actives: Vec<Vec<Active>>,
    base: Option<(Coordinates<3>, Quantity)>,
    bounds: Vec<Bound>,
    focus: Option<Vec<usize>>,
    fresh: Vec<Bound>,
    moved: Vec<Quantity<Area>>,
    values: Vec<Quantity<Area>>,
}

struct Reach<'a> {
    bounds: &'a [Bound],
    moved: &'a [Quantity<Area>],
}

impl Reach<'_> {
    fn within(&self, index: usize, hex: &[usize; 8]) -> Option<Quantity<Length>> {
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
    fn decayed(&self, index: usize, moved: Quantity<Length>) -> Bound {
        let (margin, shortest) = self.bounds[index];
        let margin = margin - moved / shortest * 12.0;
        if margin > Dimensionless::of(MARGIN) {
            (margin, shortest - moved * 2.0)
        } else {
            Default::default()
        }
    }
}

struct Energy<'a> {
    boundary: &'a [usize],
    hexes: &'a [[usize; 8]],
    kinds: &'a [u8],
    lambda: TensorListVec<Quantity<Length>, CAP>,
    planes: Vec<Plane>,
    rho: Quantity,
    scales: QuantityVector<Length>,
    scratch: Mutex<Scratch>,
    theta: Quantity,
}

impl Mesh<3> {
    pub fn improve_scaled_jacobian(
        &mut self,
        target: &Tessellation,
        threshold: Quantity,
        sweeps: usize,
    ) -> Result<(), MeshError> {
        let mut hexes = Vec::new();
        let mut kinds = Vec::new();
        for block in self.iter() {
            match block {
                Connectivity::Hexahedral(elements) => {
                    hexes.extend(elements.iter().copied());
                    kinds.resize(hexes.len(), HEXAHEDRON)
                }
                Connectivity::Tetrahedral(elements) => {
                    hexes.extend(
                        elements
                            .iter()
                            .map(|&[a, b, c, d]| [a, b, c, d, a, a, a, a]),
                    );
                    kinds.resize(hexes.len(), TETRAHEDRON)
                }
                Connectivity::Wedge(elements) => {
                    hexes.extend(
                        elements
                            .iter()
                            .map(|&[a, b, c, d, e, f]| [a, b, c, d, e, f, a, a]),
                    );
                    kinds.resize(hexes.len(), WEDGE)
                }
                Connectivity::Pyramidal(elements) => {
                    hexes.extend(
                        elements
                            .iter()
                            .map(|&[a, b, c, d, e]| [a, b, c, d, e, a, a, a]),
                    );
                    kinds.resize(hexes.len(), PYRAMID)
                }
                _ => return Err(MeshError::HexahedraRequired),
            }
        }
        let mut boundary: Vec<usize> = self.exterior_faces().into_iter().flatten().collect();
        boundary.sort_unstable();
        boundary.dedup();
        let coordinates = &mut self.coordinates;
        let surface = Surface::new(target);
        let mut energy = Energy {
            boundary: &boundary,
            hexes: &hexes,
            kinds: &kinds,
            lambda: (0..boundary.len())
                .map(|_| [Quantity::default(); CAP].into())
                .collect(),
            planes: Vec::new(),
            rho: RHO,
            scales: QuantityVector::zero(0),
            scratch: Mutex::default(),
            theta: Quantity::default(),
        };
        let mut current = coordinates.clone();
        let mut best: Option<Coordinates<3>> = None;
        let mut first = true;
        let mut corrections = Corrections::default();
        let mut nodes = Vec::<usize>::new();
        let stop = 'ratchet: loop {
            let mut stalled = false;
            let mut sweep = 0;
            loop {
                energy.refresh(&surface, &current);
                let quality = energy.quality(&current);
                let gap = energy.gap(&current);
                if first {
                    first = false;
                    if gap <= TOLERANCE {
                        best = Some(current.clone());
                        if quality >= threshold {
                            break 'ratchet None;
                        }
                        energy.theta = Self::next(quality);
                    }
                }
                if quality >= energy.theta && gap <= TOLERANCE {
                    best = Some(current.clone());
                    if quality >= threshold {
                        break 'ratchet None;
                    }
                    energy.theta = Self::next(quality);
                    continue 'ratchet;
                }
                if stalled {
                    break 'ratchet Some(MeshError::Stall {
                        theta: energy.theta.value(),
                        sweeps: sweep,
                    });
                }
                if sweep == sweeps {
                    break 'ratchet Some(MeshError::Budget {
                        theta: energy.theta.value(),
                        sweeps,
                    });
                }
                let mut active = energy.active(&current, DELTA);
                let kept = corrections.nonzero();
                if kept.len() == 3 * nodes.len() {
                    nodes
                        .iter()
                        .enumerate()
                        .filter(|(position, _)| (0..3).any(|i| kept[3 * position + i]))
                        .for_each(|(_, &node)| active[node] = true)
                }
                let next: Vec<usize> = (0..active.len()).filter(|&node| active[node]).collect();
                let mut position = vec![usize::MAX; active.len()];
                next.iter()
                    .enumerate()
                    .for_each(|(index, &node)| position[node] = index);
                corrections.remap(
                    &nodes
                        .iter()
                        .flat_map(|&node| {
                            let index = position[node];
                            (0..3).map(move |i| match index {
                                usize::MAX => usize::MAX,
                                index => 3 * index + i,
                            })
                        })
                        .collect::<Vec<_>>(),
                    3 * next.len(),
                );
                nodes = next;
                energy.focus(Some(nodes.clone()));
                let full = RefCell::new(current.clone());
                let put = |full: &mut Coordinates<3>, x: &Coordinates<3>| {
                    nodes
                        .iter()
                        .zip(x.iter())
                        .for_each(|(&node, entry)| full[node] = entry.clone())
                };
                let descent = solver(FIRST * energy.typical())
                    .descend_from(
                        &mut corrections,
                        |x: &Coordinates<3>| {
                            let mut full = full.borrow_mut();
                            put(&mut full, x);
                            Ok(energy.evaluate(&full, None))
                        },
                        |x: &Coordinates<3>| {
                            let mut full = full.borrow_mut();
                            put(&mut full, x);
                            let mut slopes = Gradient::zero(full.len());
                            energy.evaluate(&full, Some(&mut slopes));
                            Ok(nodes
                                .iter()
                                .map(|&node| slopes[node].clone())
                                .collect::<Gradient>())
                        },
                        nodes
                            .iter()
                            .map(|&node| current[node].clone())
                            .collect::<Coordinates<3>>(),
                        Vec::new(),
                    )
                    .map_err(|_| MeshError::Minimization)?;
                let decrease = (descent.initial - descent.value)
                    / descent
                        .initial
                        .abs()
                        .max(Quantity::new(Scalar::MIN_POSITIVE));
                energy.focus(None);
                nodes
                    .iter()
                    .zip(descent.solution.iter())
                    .for_each(|(&node, entry)| current[node] = entry.clone());
                energy.settle(&current);
                sweep += 1;
                if decrease < STALL {
                    stalled = true;
                }
            }
        };
        *coordinates = best.ok_or(MeshError::SurfaceFit)?;
        stop.map_or(Ok(()), Err)
    }
    fn next(quality: Quantity) -> Quantity {
        ((quality / STEP).floor() + Dimensionless::of(1.0)) * STEP
    }
}

impl Plane {
    fn residual(&self, position: &Coordinate<3>) -> Quantity<Length> {
        position * &self.normal - self.offset
    }
}

struct Surface<'a> {
    normals: DirectionsRef<'a, 3>,
    surface: &'a Coordinates<3>,
    elements: Vec<&'a [usize]>,
    target: &'a Tessellation,
}

impl<'a> Surface<'a> {
    fn new(target: &'a Tessellation) -> Self {
        let mesh = target.mesh();
        Self {
            normals: target.normals().iter().flatten().collect(),
            surface: mesh.coordinates(),
            elements: mesh.connectivities().iter().flatten().collect(),
            target,
        }
    }
    fn planes(
        &self,
        boundary: &[usize],
        sizes: &QuantityVector<Length>,
        coordinates: &Coordinates<3>,
    ) -> Vec<Plane> {
        let bvh = self.target.bvh();
        let (surface, elements, normals) = (self.surface, &self.elements, &self.normals);
        let chunk = chunk_size(boundary.len());
        scope(|scope| {
            boundary
                .chunks(chunk)
                .enumerate()
                .map(|(block, nodes)| {
                    scope.spawn(move || {
                        let mut planes = Vec::new();
                        nodes.iter().enumerate().for_each(|(offset, &node)| {
                            let index = block * chunk + offset;
                            let position = &coordinates[node];
                            let ties =
                                bvh.closest_points(position, surface, elements, sizes[node] * TIE);
                            let Some((point, _)) = ties.first().cloned() else {
                                return;
                            };
                            let tied: Vec<usize> =
                                ties.iter().map(|&(_, triangle)| triangle).collect();
                            let reach = sizes[node] * RADIUS;
                            let (low, high) = (
                                from_fn(|i| point[i] - reach).into(),
                                from_fn(|i| point[i] + reach).into(),
                            );
                            let query: BoundingBox<3> = [&low, &high]
                                .into_iter()
                                .collect::<CoordinatesRef<'_, 3>>()
                                .into();
                            let mut kept = Vec::<Direction<3>>::new();
                            for triangle in tied.iter().copied().chain(bvh.overlapping(&query)) {
                                if kept.len() == CAP {
                                    break;
                                }
                                let nodes = elements[triangle];
                                let tie = tied.contains(&triangle);
                                if !tie
                                    && !query.overlaps_triangle(
                                        &surface[nodes[0]],
                                        &surface[nodes[1]],
                                        &surface[nodes[2]],
                                    )
                                {
                                    continue;
                                }
                                let normal = normals[triangle].clone();
                                if kept.iter().any(|other| (other * &normal).abs() > CREASE) {
                                    continue;
                                }
                                let plane = Plane {
                                    boundary: index,
                                    offset: &surface[elements[triangle][0]] * &normal,
                                    normal,
                                    slot: kept.len(),
                                };
                                if !tie && plane.residual(position).abs() > sizes[node] * TOLERANCE
                                {
                                    continue;
                                }
                                kept.push(plane.normal.clone());
                                planes.push(plane)
                            }
                        });
                        planes
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect()
        })
    }
}

impl Energy<'_> {
    fn refresh(&mut self, surface: &Surface, coordinates: &Coordinates<3>) {
        self.scales = scales(self.hexes, self.kinds, coordinates);
        let sizes = node_scales(self.boundary, self.hexes, &self.scales, coordinates.len());
        self.planes = surface.planes(self.boundary, &sizes, coordinates);
    }
    fn typical(&self) -> Quantity<Length> {
        self.scales.iter().copied().sum::<Quantity<Length>>() / self.scales.len().max(1) as Scalar
    }
    fn focus(&self, nodes: Option<Vec<usize>>) {
        self.scratch.lock().unwrap().focus = nodes
    }
    fn quality(&self, coordinates: &Coordinates<3>) -> Quantity {
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
    fn active(&self, coordinates: &Coordinates<3>, delta: Scalar) -> Vec<bool> {
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
    fn gap(&self, coordinates: &Coordinates<3>) -> Quantity {
        let sizes = node_scales(self.boundary, self.hexes, &self.scales, coordinates.len());
        self.planes
            .iter()
            .map(|plane| {
                let node = self.boundary[plane.boundary];
                plane.residual(&coordinates[node]).abs() / sizes[node]
            })
            .fold(Quantity::default(), Quantity::max)
    }
    fn settle(&mut self, coordinates: &Coordinates<3>) {
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
    fn evaluate(
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

fn displacements(
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

fn chunk_size(items: usize) -> usize {
    if items < 2 * SERIAL {
        items.max(1)
    } else {
        items
            .div_ceil(available_parallelism().map_or(1, |threads| threads.get()))
            .max(1)
    }
}

fn scales(
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

fn node_scales(
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
