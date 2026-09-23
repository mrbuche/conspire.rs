#[cfg(test)]
mod test;

use super::Partition;
use crate::geometry::mesh::Mesh;
use std::{array::from_fn, cmp::Ordering};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bisection {
    Coordinate,
    Principal,
}

impl<const D: usize> Mesh<D> {
    pub fn partition_rcb(&self, parts: usize) -> Partition {
        self.partition_bisection(parts, Bisection::Coordinate)
    }
    pub fn partition_rib(&self, parts: usize) -> Partition {
        self.partition_bisection(parts, Bisection::Principal)
    }
    pub fn partition_bisection(&self, parts: usize, bisection: Bisection) -> Partition {
        let points = self.element_points();
        assert!(
            (1..=points.len()).contains(&parts),
            "parts must be between 1 and the number of elements"
        );
        let mut assignment = vec![0; points.len()];
        let mut elements = (0..points.len()).collect::<Vec<_>>();
        bisect(&mut elements, parts, 0, &points, bisection, &mut assignment);
        Partition::new(self, assignment)
    }
}

fn bisect<const D: usize>(
    elements: &mut [usize],
    parts: usize,
    first: usize,
    points: &[[f64; D]],
    bisection: Bisection,
    assignment: &mut [usize],
) {
    if parts == 1 {
        elements
            .iter()
            .for_each(|&element| assignment[element] = first);
        return;
    }
    let lower = parts / 2;
    let split = elements.len() * lower / parts;
    let direction = match bisection {
        Bisection::Coordinate => widest_axis(elements, points),
        Bisection::Principal => principal_axis(elements, points),
    };
    let key = |element: usize| -> f64 {
        (0..D)
            .map(|axis| direction[axis] * points[element][axis])
            .sum()
    };
    elements.select_nth_unstable_by(split, |&a, &b| {
        key(a)
            .partial_cmp(&key(b))
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    let (left, right) = elements.split_at_mut(split);
    bisect(left, lower, first, points, bisection, assignment);
    bisect(
        right,
        parts - lower,
        first + lower,
        points,
        bisection,
        assignment,
    );
}

fn widest_axis<const D: usize>(elements: &[usize], points: &[[f64; D]]) -> [f64; D] {
    let extent = |axis: usize| {
        let (low, high) = elements.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(low, high), &element| {
                (
                    low.min(points[element][axis]),
                    high.max(points[element][axis]),
                )
            },
        );
        high - low
    };
    let widest = (1..D).fold(0, |best, axis| {
        if extent(axis) > extent(best) {
            axis
        } else {
            best
        }
    });
    from_fn(|index| if index == widest { 1.0 } else { 0.0 })
}

fn principal_axis<const D: usize>(elements: &[usize], points: &[[f64; D]]) -> [f64; D] {
    let count = elements.len() as f64;
    let mean: [f64; D] = from_fn(|axis| {
        elements
            .iter()
            .map(|&element| points[element][axis])
            .sum::<f64>()
            / count
    });
    let mut covariance = [[0.0; D]; D];
    elements.iter().for_each(|&element| {
        (0..D).for_each(|row| {
            (0..D).for_each(|column| {
                covariance[row][column] +=
                    (points[element][row] - mean[row]) * (points[element][column] - mean[column])
            })
        })
    });
    let mut direction: [f64; D] = from_fn(|axis| 1.0 + 0.1 * axis as f64);
    (0..64).for_each(|_| {
        let next: [f64; D] = from_fn(|row| {
            (0..D)
                .map(|column| covariance[row][column] * direction[column])
                .sum()
        });
        let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            direction = from_fn(|axis| next[axis] / norm);
        }
    });
    if direction.iter().all(|x| x.is_finite()) && direction.iter().any(|&x| x != 0.0) {
        direction
    } else {
        widest_axis(elements, points)
    }
}
