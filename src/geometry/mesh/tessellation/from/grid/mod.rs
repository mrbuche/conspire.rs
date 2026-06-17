#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivity, Mesh, tessellation::Tessellation},
    },
    math::{Tensor, TensorVec},
};
use std::collections::HashMap;

const FACES: [([isize; 3], [[usize; 3]; 4]); 6] = [
    ([1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
    ([-1, 0, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
    ([0, 1, 0], [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
    ([0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
    ([0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
    ([0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
];

impl<T> From<Voxels<T>> for Tessellation
where
    T: Copy + Default + Ord,
{
    fn from(voxels: Voxels<T>) -> Self {
        let [nx, ny, nz] = *voxels.nel();
        let (nxp, nyp) = (nx + 1, ny + 1);
        let void = T::default();
        let data = voxels.data();
        let mut vertices = HashMap::new();
        let mut coordinates = Coordinates::new();
        let mut triangles: Vec<[usize; 3]> = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let a = data[i + nx * j + nx * ny * k];
                    for (delta, corners) in &FACES {
                        let neighbor = [
                            i as isize + delta[0],
                            j as isize + delta[1],
                            k as isize + delta[2],
                        ];
                        let b = if neighbor
                            .iter()
                            .zip([nx, ny, nz])
                            .any(|(&n, m)| n < 0 || n >= m as isize)
                        {
                            void
                        } else {
                            data[neighbor[0] as usize
                                + nx * neighbor[1] as usize
                                + nx * ny * neighbor[2] as usize]
                        };
                        if a > b {
                            let quad = corners.map(|off| {
                                vertex(&mut vertices, &mut coordinates, [nxp, nyp], [
                                    i + off[0],
                                    j + off[1],
                                    k + off[2],
                                ])
                            });
                            triangles.push([quad[0], quad[1], quad[2]]);
                            triangles.push([quad[0], quad[2], quad[3]]);
                        }
                    }
                }
            }
        }
        let connectivities = vec![Connectivity::Triangular(triangles.into())];
        Tessellation::from(Mesh::from((connectivities, coordinates)))
    }
}

fn vertex(
    vertices: &mut HashMap<usize, usize>,
    coordinates: &mut Coordinates<3>,
    [nxp, nyp]: [usize; 2],
    corner: [usize; 3],
) -> usize {
    *vertices
        .entry(corner[0] + nxp * (corner[1] + nyp * corner[2]))
        .or_insert_with(|| {
            let index = coordinates.len();
            coordinates.push(Coordinate::const_from([
                corner[0] as f64,
                corner[1] as f64,
                corner[2] as f64,
            ]));
            index
        })
}
