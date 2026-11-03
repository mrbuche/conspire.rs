#[cfg(test)]
mod test;

use super::{
    MARGIN,
    kind::{Kind, Point, Weights},
};
use crate::{
    geometry::{Coordinate, CoordinateList, DirectionList},
    math::{CrossProduct, Quantity, Reference, Scalar, Tensor, TensorList, TensorRank1},
    units::{Area, Dimensionless, Length, ReciprocalLength, Volume},
};
use std::array::from_fn;

pub(super) struct Frame {
    pub(super) edges: CoordinateList<3, 3>,
    pub(super) jacobian: Quantity<Volume>,
    pub(super) lengths: TensorList<Quantity<Length>, 3>,
    pub(super) scaled: Quantity,
}

pub(super) type Bound = (Quantity, Quantity<Length>);

pub(super) type Part = (Quantity<Area>, CoordinateList<3, 8>, Bound);

impl Frame {
    fn vacant() -> Self {
        Self {
            edges: CoordinateList::default(),
            jacobian: Quantity::new(Scalar::INFINITY),
            lengths: from_fn(|_| Quantity::new(Scalar::INFINITY)).into(),
            scaled: Quantity::new(Scalar::INFINITY),
        }
    }
    pub(super) fn new(point: &Point, nodes: &CoordinateList<3, 8>, normalization: Scalar) -> Self {
        let edges: CoordinateList<3, 3> = point
            .sparse
            .iter()
            .map(|terms| {
                terms
                    .iter()
                    .fold(Coordinate::default(), |sum, &(node, weight)| {
                        sum + &nodes[node] * weight
                    })
            })
            .collect();
        let jacobian = &edges[0] * edges[1].cross(&edges[2]);
        let lengths: TensorList<Quantity<Length>, 3> =
            edges.iter().map(|edge| edge.norm()).collect();
        let product = lengths[0] * lengths[1] * lengths[2];
        let scaled = if lengths.iter().all(|&length| length > Quantity::default()) {
            jacobian / product * normalization
        } else {
            Quantity::default()
        };
        Self {
            edges,
            jacobian,
            lengths,
            scaled,
        }
    }
}

pub(super) fn bounded(
    kind: &Kind,
    theta: Quantity,
    scale: Quantity<Length>,
    nodes: &CoordinateList<3, 8>,
    derivative: bool,
) -> Part {
    let frames: [Frame; 9] = from_fn(|point| {
        kind.points.get(point).map_or_else(Frame::vacant, |point| {
            Frame::new(point, nodes, kind.normalization)
        })
    });
    let (j, lowest_j) = frames
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.jacobian.total_cmp(&b.1.jacobian))
        .map(|(point, frame)| (point, frame.jacobian))
        .unwrap();
    let (s, lowest_sj) = frames
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.scaled.total_cmp(&b.1.scaled))
        .map(|(point, frame)| (point, frame.scaled))
        .unwrap();
    let mut slopes = CoordinateList::default();
    if lowest_j <= Quantity::default() {
        if derivative {
            let e = &frames[j].edges;
            let g = [
                (e[1].cross(&e[2]) * kind.normalization / scale).with_unit(),
                (e[2].cross(&e[0]) * kind.normalization / scale).with_unit(),
                (e[0].cross(&e[1]) * kind.normalization / scale).with_unit(),
            ]
            .into();
            scatter(&kind.points[j].weights, &g, &mut slopes)
        }
        (
            lowest_j * kind.normalization / scale - theta * scale * scale,
            slopes,
            Default::default(),
        )
    } else if lowest_sj <= theta {
        let area = scale * scale;
        if derivative {
            let frame = &frames[s];
            let u: DirectionList<3, 3> = frame
                .edges
                .iter()
                .map(|edge| edge.clone().normalized())
                .collect();
            let g = (0..3)
                .map(|i| {
                    let c = u[(i + 1) % 3].cross(&u[(i + 2) % 3]);
                    let along = &c * &u[i];
                    let slope: TensorRank1<3, Reference, ReciprocalLength> =
                        ((c - &u[i] * along) / frame.lengths[i]).with_unit();
                    (slope * area * kind.normalization).with_unit()
                })
                .collect();
            scatter(&kind.points[s].weights, &g, &mut slopes)
        }
        ((lowest_sj - theta) * area, slopes, Default::default())
    } else {
        let margin = lowest_sj - theta.max(Quantity::default());
        let bound = if derivative && margin > Dimensionless::of(MARGIN) {
            let shortest = frames
                .iter()
                .flat_map(|frame| frame.lengths.iter().copied())
                .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
            (margin, shortest)
        } else {
            Default::default()
        };
        (Quantity::default(), slopes, bound)
    }
}

fn scatter(weights: &Weights, g: &CoordinateList<3, 3>, slopes: &mut CoordinateList<3, 8>) {
    weights.iter().zip(g).for_each(|(weight_i, g_i)| {
        slopes
            .iter_mut()
            .zip(weight_i)
            .for_each(|(slope, &weight)| *slope += g_i * weight)
    })
}
