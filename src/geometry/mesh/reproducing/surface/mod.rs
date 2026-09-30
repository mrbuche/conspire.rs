#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Basis, Connectivity, Mesh},
    math::{Quantity, Reference, Tensor, TensorRank1},
    units::{Area, Length},
};
use std::array::from_fn;

const NOT_TRIANGULAR: &str = "surface integrals require triangular faces";
const NOT_ON_THE_BOUNDARY: &str = "surface normals require faces on the boundary of the mesh";

type Point = [f64; 3];

fn points(mesh: &Mesh<3>) -> Vec<Point> {
    mesh.coordinates()
        .iter()
        .map(|x| std::array::from_fn(|k| x[k].value_as::<Length>()))
        .collect()
}

fn triangle(face: &[usize]) -> Result<[usize; 3], &'static str> {
    face.try_into().map_err(|_| NOT_TRIANGULAR)
}

fn area_vector(points: &[Point], [a, b, c]: [usize; 3]) -> Point {
    let u: Point = from_fn(|k| points[b][k] - points[a][k]);
    let v: Point = from_fn(|k| points[c][k] - points[a][k]);
    [
        0.5 * (u[1] * v[2] - u[2] * v[1]),
        0.5 * (u[2] * v[0] - u[0] * v[2]),
        0.5 * (u[0] * v[1] - u[1] * v[0]),
    ]
}

fn centroid(points: &[Point], nodes: &[usize]) -> Point {
    from_fn(|k| nodes.iter().map(|&n| points[n][k]).sum::<f64>() / nodes.len() as f64)
}

fn area_shares(mesh: &Mesh<3>, faces: &[Vec<usize>]) -> Result<Vec<f64>, &'static str> {
    let points = points(mesh);
    let mut shares = vec![0.0; points.len()];
    for face in faces {
        let face = triangle(face)?;
        let vector = area_vector(&points, face);
        let area = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        face.iter().for_each(|&node| shares[node] += area / 3.0);
    }
    Ok(shares)
}

fn vector_shares(mesh: &Mesh<3>, faces: &[Vec<usize>]) -> Result<Vec<Point>, &'static str> {
    let points = points(mesh);
    let elements: Vec<(&Connectivity, &[usize])> = mesh
        .iter()
        .flat_map(|block| block.iter().map(move |element| (block, element)))
        .collect();
    let mut shares = vec![[0.0; 3]; points.len()];
    for face in faces {
        let face = triangle(face)?;
        let owners: Vec<Vec<usize>> = mesh.node_element_connectivity()[face[0]]
            .iter()
            .map(|&element| elements[element].0.element_nodes(elements[element].1))
            .filter(|nodes| face.iter().all(|node| nodes.contains(node)))
            .collect();
        let [owner] = &owners[..] else {
            return Err(NOT_ON_THE_BOUNDARY);
        };
        let mut vector = area_vector(&points, face);
        let outward: Point =
            std::array::from_fn(|k| centroid(&points, &face)[k] - centroid(&points, owner)[k]);
        if (0..3).map(|k| vector[k] * outward[k]).sum::<f64>() < 0.0 {
            vector = vector.map(|x| -x);
        }
        for node in face {
            (0..3).for_each(|k| shares[node][k] += vector[k] / 3.0);
        }
    }
    Ok(shares)
}

impl Mesh<3> {
    /// The integral of each basis function over triangular faces of the mesh,
    /// exact for its linear interpolation over them.
    pub fn face_integrals(
        &self,
        basis: &Basis,
        faces: &[Vec<usize>],
    ) -> Result<Vec<Quantity<Area>>, &'static str> {
        let shares = area_shares(self, faces)?;
        Ok(basis
            .values
            .iter()
            .map(|values| {
                Quantity::new(
                    values
                        .iter()
                        .map(|&(node, value)| value * shares[node])
                        .sum(),
                )
            })
            .collect())
    }
    /// The integral of each basis function times the outward normal over
    /// triangular faces on the boundary of the mesh.
    ///
    /// Over all the exterior faces, this is the integral over the mesh of the
    /// gradient of each basis function.
    pub fn face_normal_integrals(
        &self,
        basis: &Basis,
        faces: &[Vec<usize>],
    ) -> Result<Vec<TensorRank1<3, Reference, Area>>, &'static str> {
        let shares = vector_shares(self, faces)?;
        Ok(basis
            .values
            .iter()
            .map(|values| {
                let mut sum = [0.0; 3];
                for &(node, value) in values {
                    (0..3).for_each(|k| sum[k] += value * shares[node][k]);
                }
                TensorRank1::from(sum)
            })
            .collect())
    }
}
