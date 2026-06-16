#[cfg(test)]
mod test;

use crate::geometry::{Coordinate, grid::Voxels, mesh::Tessellation};
use std::{
    array::from_fn,
    thread::{available_parallelism, scope},
};

impl Voxels<usize> {
    pub fn from_tessellation(tessellation: &Tessellation, size: f64) -> Self {
        let mesh = tessellation.mesh();
        let bvh = tessellation.bvh();
        let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
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
        let [nx, ny, _] = nel;
        let layer = nx * ny;
        let mut data = vec![0usize; layer * nel[2]];
        let direction = Coordinate::from([1.0, 0.01, 0.001]);
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = data.len().div_ceil(threads).max(1);
        scope(|scope| {
            let (elements, direction) = (&elements, &direction);
            data.chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, voxels)| {
                    let offset = chunk * chunk_size;
                    scope.spawn(move || {
                        voxels.iter_mut().enumerate().for_each(|(local, voxel)| {
                            let flat = offset + local;
                            let index = [flat % nx, flat / nx % ny, flat / layer];
                            let center = Coordinate::from(from_fn::<_, 3, _>(|ax| {
                                min[ax] + (index[ax] as f64 + 0.5) * size
                            }));
                            let ray = (center, direction.clone()).into();
                            if bvh.intersections(&ray, coordinates, elements) % 2 == 1 {
                                *voxel = 1;
                            }
                        });
                    });
                });
        });
        Voxels::new(data, nel)
    }
}
