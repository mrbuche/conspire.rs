#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            remesh::triangles::remesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{Scalar, Tensor},
};
use std::{
    array::from_fn,
    thread::{available_parallelism, scope},
};

const STITCH_TRIM_MARGIN: Scalar = 1.0;

impl Tessellation {
    pub fn fitted_surface(
        &self,
        balancing: Balancing,
        scale: Scalar,
        curvature: CurvatureSizing,
        iterations: usize,
    ) -> Result<(Mesh<D>, Mesh<D>), &'static str> {
        let mut octree = Octree::<u16, usize>::from_features(self, scale, curvature, 0);
        octree.equilibrate(balancing, Pairing::Regular)?;
        let mut core = octree.dualize();
        self.trim(&mut core, self.bvh(), STITCH_TRIM_MARGIN);
        let sizing = QuadSizing::new(&core)?;
        let elements: Vec<&[usize]> = self.mesh().connectivities().iter().flatten().collect();
        let mut connectivity: Vec<[usize; 3]> = elements
            .iter()
            .map(|element| from_fn(|i| element[i]))
            .collect();
        let mut coordinates = self.mesh().coordinates().clone();
        remesh(
            &mut connectivity,
            &mut coordinates,
            iterations,
            |_, points, _| sizing.target_lengths(points),
        )?;
        let surface = (vec![connectivity.into()], coordinates).into();
        Ok((core, surface))
    }
}

struct QuadSizing {
    bvh: BoundingVolumeHierarchy<D>,
    coordinates: Coordinates<D>,
    triangles: Vec<[usize; 3]>,
    lengths: Vec<Scalar>,
}

impl QuadSizing {
    fn new(core: &Mesh<D>) -> Result<Self, &'static str> {
        let coordinates = core.coordinates();
        let quads = core.exterior_faces();
        if quads.iter().any(|quad| quad.len() != 4) {
            return Err("fitted_surface requires an all-hexahedral trimmed core");
        }
        let lengths: Vec<Scalar> = quads
            .iter()
            .map(|quad| {
                (0..4)
                    .map(|i| (&coordinates[quad[i]] - &coordinates[quad[(i + 1) % 4]]).norm())
                    .sum::<Scalar>()
                    / 4.0
            })
            .collect();
        let triangles: Vec<[usize; 3]> = quads
            .iter()
            .flat_map(|quad| [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
            .collect();
        let mesh: Mesh<D> = (
            vec![Connectivity::Triangular(triangles.clone().into())],
            coordinates.clone(),
        )
            .into();
        let bvh = BoundingVolumeHierarchy::from(&mesh);
        Ok(Self {
            bvh,
            coordinates: coordinates.clone(),
            triangles,
            lengths,
        })
    }
    fn target_lengths(&self, points: &Coordinates<D>) -> Vec<Scalar> {
        let elements: Vec<&[usize]> = self.triangles.iter().map(|t| t.as_slice()).collect();
        let number_of_points = points.len();
        let mut targets = vec![Scalar::INFINITY; number_of_points];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_points.div_ceil(threads).max(1);
        scope(|scope| {
            let (bvh, coordinates, elements, lengths) =
                (&self.bvh, &self.coordinates, &elements, &self.lengths);
            targets
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, out)| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        out.iter_mut().enumerate().for_each(|(local, target)| {
                            let point = &points[offset + local];
                            if let Some((_, triangle)) =
                                bvh.closest_point(point, coordinates, elements)
                            {
                                *target = lengths[triangle / 2];
                            }
                        });
                    });
                });
        });
        targets
    }
}
