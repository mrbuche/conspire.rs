#[cfg(test)]
mod test;

mod element;
mod energy;
mod kind;
mod surface;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, MeshError, Tessellation},
    },
    math::{
        Quantity, QuantityVector, Reference, Scalar, Tensor, TensorRank1Vec,
        optimize::{Armijo, Corrections, LimitedMemoryBFGS, Tolerances},
    },
    units::{Dimensionless, Length},
};
use energy::Energy;
use kind::{HEXAHEDRON, PYRAMID, TETRAHEDRON, WEDGE};
use std::{cell::RefCell, sync::Mutex, thread::available_parallelism};
use surface::Surface;

type Gradient = TensorRank1Vec<3, Reference, Length>;

const CAP: usize = 4;
const DELTA: Scalar = 0.05;
const SERIAL: usize = 512;
const FIRST: Quantity = Dimensionless::of(0.01);
const MARGIN: Scalar = 1.0e-9;
const RHO: Quantity = Dimensionless::of(1.0e2);
const STALL: Quantity = Dimensionless::of(1.0e-6);
const STEP: Quantity = Dimensionless::of(0.01);
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

fn chunk_size(items: usize) -> usize {
    if items < 2 * SERIAL {
        items.max(1)
    } else {
        items
            .div_ceil(available_parallelism().map_or(1, |threads| threads.get()))
            .max(1)
    }
}
