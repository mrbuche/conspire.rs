#[cfg(test)]
mod test;

use crate::geometry::{grid::Voxels, mesh::Mesh};
use std::array::from_fn;

const TETS_4: [[usize; 4]; 1] = [[0, 1, 2, 3]];
const TETS_5: [[usize; 4]; 2] = [[0, 1, 2, 4], [0, 2, 3, 4]];
const TETS_6: [[usize; 4]; 3] = [[0, 1, 2, 3], [1, 4, 5, 3], [1, 2, 5, 3]];
const TETS_8: [[usize; 4]; 6] = [
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
    [0, 5, 1, 6],
];

impl Voxels<usize> {
    pub fn from_finite_elements(mesh: &Mesh<3>, size: f64) -> Self {
        let coordinates = mesh.coordinates();
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for point in coordinates {
            (0..3).for_each(|ax| {
                min[ax] = min[ax].min(point[ax]);
                max[ax] = max[ax].max(point[ax]);
            });
        }
        let nel: [usize; 3] = from_fn(|ax| (((max[ax] - min[ax]) / size).ceil() as usize).max(1));
        let [nx, ny, nz] = nel;
        let mut data = vec![0usize; nx * ny * nz];
        let numbers = mesh.blocks();
        for (block, connectivity) in mesh.iter().enumerate() {
            let material = numbers.map_or(block + 1, |numbers| numbers[block]);
            for nodes in connectivity {
                let tets: &[[usize; 4]] = match nodes.len() {
                    4 => &TETS_4,
                    5 => &TETS_5,
                    6 => &TETS_6,
                    8 => &TETS_8,
                    _ => continue,
                };
                let points: Vec<[f64; 3]> = nodes
                    .iter()
                    .map(|&node| from_fn(|ax| coordinates[node][ax]))
                    .collect();
                let mut lo = [usize::MAX; 3];
                let mut hi = [0usize; 3];
                for point in &points {
                    (0..3).for_each(|ax| {
                        let l = ((point[ax] - min[ax]) / size).floor().max(0.0) as usize;
                        let h = (((point[ax] - min[ax]) / size).floor() as usize + 1).min(nel[ax]);
                        lo[ax] = lo[ax].min(l);
                        hi[ax] = hi[ax].max(h);
                    });
                }
                for k in lo[2]..hi[2] {
                    for j in lo[1]..hi[1] {
                        for i in lo[0]..hi[0] {
                            let center =
                                from_fn(|ax| min[ax] + ([i, j, k][ax] as f64 + 0.5) * size);
                            if tets
                                .iter()
                                .any(|tet| inside(center, from_fn(|vertex| points[tet[vertex]])))
                            {
                                data[i + nx * j + nx * ny * k] = material;
                            }
                        }
                    }
                }
            }
        }
        Voxels::new(data, nel)
    }
}

fn inside(query: [f64; 3], tet: [[f64; 3]; 4]) -> bool {
    let volume = orient(tet[0], tet[1], tet[2], tet[3]);
    if volume == 0.0 {
        return false;
    }
    let sign = volume.signum();
    let tolerance = -1e-9 * volume.abs();
    sign * orient(query, tet[1], tet[2], tet[3]) >= tolerance
        && sign * orient(tet[0], query, tet[2], tet[3]) >= tolerance
        && sign * orient(tet[0], tet[1], query, tet[3]) >= tolerance
        && sign * orient(tet[0], tet[1], tet[2], query) >= tolerance
}

fn orient(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u = from_fn::<_, 3, _>(|ax| b[ax] - a[ax]);
    let v = from_fn::<_, 3, _>(|ax| c[ax] - a[ax]);
    let w = from_fn::<_, 3, _>(|ax| d[ax] - a[ax]);
    u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
}
