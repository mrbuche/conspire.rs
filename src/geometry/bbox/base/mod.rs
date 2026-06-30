#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, bbox::BoundingBox},
    math::{Scalar, Tensor},
};
use std::array::from_fn;

impl<const D: usize> BoundingBox<D> {
    pub fn minimum(&self) -> &Coordinate<D> {
        &self.minimum
    }
    pub fn maximum(&self) -> &Coordinate<D> {
        &self.maximum
    }
    pub fn overlaps(&self, other: &Self) -> bool {
        (0..D).all(|d| self.minimum[d] <= other.maximum[d] && other.minimum[d] <= self.maximum[d])
    }
    pub fn longest_axis(&self) -> usize {
        self.maximum
            .iter()
            .zip(self.minimum.iter())
            .enumerate()
            .map(|(i, (&max, &min))| (i, max - min))
            .max_by(|(_, length_a), (_, length_b)| length_a.partial_cmp(length_b).unwrap())
            .unwrap()
            .0
    }
    pub fn shortest_axis(&self) -> usize {
        self.maximum
            .iter()
            .zip(self.minimum.iter())
            .enumerate()
            .map(|(i, (&max, &min))| (i, max - min))
            .min_by(|(_, length_a), (_, length_b)| length_a.partial_cmp(length_b).unwrap())
            .unwrap()
            .0
    }
}

impl BoundingBox<3> {
    pub fn overlaps_triangle(
        &self,
        a: &Coordinate<3>,
        b: &Coordinate<3>,
        c: &Coordinate<3>,
    ) -> bool {
        let center: [Scalar; 3] = from_fn(|k| (self.minimum[k] + self.maximum[k]) * 0.5);
        let half: [Scalar; 3] = from_fn(|k| (self.maximum[k] - self.minimum[k]) * 0.5);
        let v: [[Scalar; 3]; 3] = [a, b, c].map(|p| from_fn(|k| p[k] - center[k]));
        let edges: [[Scalar; 3]; 3] = [
            from_fn(|k| v[1][k] - v[0][k]),
            from_fn(|k| v[2][k] - v[1][k]),
            from_fn(|k| v[0][k] - v[2][k]),
        ];
        for k in 0..3 {
            for e in &edges {
                let axis = match k {
                    0 => [0.0, -e[2], e[1]],
                    1 => [e[2], 0.0, -e[0]],
                    _ => [-e[1], e[0], 0.0],
                };
                let radius = (0..3).map(|i| half[i] * axis[i].abs()).sum();
                let projection: [Scalar; 3] = from_fn(|i| (0..3).map(|j| axis[j] * v[i][j]).sum());
                let low = projection[0].min(projection[1]).min(projection[2]);
                let high = projection[0].max(projection[1]).max(projection[2]);
                if low > radius || high < -radius {
                    return false;
                }
            }
        }
        for k in 0..3 {
            let low = v[0][k].min(v[1][k]).min(v[2][k]);
            let high = v[0][k].max(v[1][k]).max(v[2][k]);
            if low > half[k] || high < -half[k] {
                return false;
            }
        }
        let normal: [Scalar; 3] = from_fn(|k| {
            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
            edges[0][i] * edges[1][j] - edges[0][j] * edges[1][i]
        });
        let radius = (0..3).map(|i| half[i] * normal[i].abs()).sum();
        let distance: Scalar = (0..3).map(|i| normal[i] * v[0][i]).sum();
        distance.abs() <= radius
    }
}
