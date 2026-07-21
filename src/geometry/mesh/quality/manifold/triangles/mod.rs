#[cfg(test)]
mod test;

use crate::{
    ABS_TOL,
    geometry::{Coordinates, bvh::BoundingVolumeHierarchy, mesh::Mesh},
    math::{CrossProduct, Scalar},
};

const D: usize = 3;
const N: usize = 3;

impl Mesh<D> {
    pub fn self_intersections(&self) -> Vec<[usize; 2]> {
        let coordinates = self.coordinates();
        let faces: Vec<[usize; N]> = self
            .iter()
            .flat_map(|block| block.iter())
            .map(|element| [element[0], element[1], element[2]])
            .collect();
        let boxes = self.bounding_boxes();
        let bvh = BoundingVolumeHierarchy::from(self);
        let mut hits = Vec::new();
        for (i, face) in faces.iter().enumerate() {
            for j in bvh.overlapping(&boxes[i]) {
                if j > i
                    && !face.iter().any(|node| faces[j].contains(node))
                    && triangles_intersect(*face, faces[j], coordinates)
                {
                    hits.push([i, j]);
                }
            }
        }
        hits
    }
}

fn triangles_intersect(t1: [usize; N], t2: [usize; N], coordinates: &Coordinates<D>) -> bool {
    let v = t1.map(|i| &coordinates[i]);
    let u = t2.map(|i| &coordinates[i]);
    let n1 = (v[1] - v[0]).cross(v[2] - v[0]);
    let du = u.map(|p| &n1 * &(p - v[0]));
    if du[0] * du[1] > 0.0 && du[0] * du[2] > 0.0 {
        return false;
    }
    let n2 = (u[1] - u[0]).cross(u[2] - u[0]);
    let dv = v.map(|p| &n2 * &(p - u[0]));
    if dv[0] * dv[1] > 0.0 && dv[0] * dv[2] > 0.0 {
        return false;
    }
    let direction = n1.cross(&n2);
    if &direction * &direction < ABS_TOL * (&n1 * &n1) * (&n2 * &n2) {
        return false; // coplanar (or degenerate)
    }
    let axis = {
        let d = [direction[0].abs(), direction[1].abs(), direction[2].abs()];
        if d[0] >= d[1] && d[0] >= d[2] {
            0
        } else if d[1] >= d[2] {
            1
        } else {
            2
        }
    };
    let interval1 = interval([v[0][axis], v[1][axis], v[2][axis]], dv);
    let interval2 = interval([u[0][axis], u[1][axis], u[2][axis]], du);
    interval1[0] <= interval2[1] && interval2[0] <= interval1[1]
}

fn interval(projection: [Scalar; N], distance: [Scalar; N]) -> [Scalar; 2] {
    let (pivot, a, b) = if distance[0] * distance[1] > 0.0 {
        (2, 0, 1)
    } else if distance[0] * distance[2] > 0.0 {
        (1, 0, 2)
    } else if distance[1] * distance[2] > 0.0 || distance[0] != 0.0 {
        (0, 1, 2)
    } else if distance[1] != 0.0 {
        (1, 0, 2)
    } else {
        (2, 0, 1)
    };
    let crossing = |q| {
        projection[pivot]
            + (projection[q] - projection[pivot]) * distance[pivot]
                / (distance[pivot] - distance[q])
    };
    let (ta, tb) = (crossing(a), crossing(b));
    if ta <= tb { [ta, tb] } else { [tb, ta] }
}
