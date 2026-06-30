#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            smooth::Smoothing,
            tessellation::{D, Normals, Tessellation},
        },
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::{array::from_fn, cell::OnceCell, collections::HashMap};

const WELD_TOLERANCE: Scalar = 1e-6;

impl Tessellation {
    pub fn mesh(&self) -> &Mesh<D> {
        &self.mesh
    }
    pub fn normals(&self) -> &Normals {
        &self.normals
    }
    pub fn bvh(&self) -> &BoundingVolumeHierarchy<D> {
        self.bvh
            .get_or_init(|| BoundingVolumeHierarchy::from(&self.mesh))
    }
    pub fn smooth(&mut self, smoothing: Smoothing) {
        self.mesh.smooth(smoothing);
        self.refresh();
    }
    pub fn smooth_welded(&mut self, smoothing: Smoothing) {
        let mut min = [f64::INFINITY; D];
        let mut max = [f64::NEG_INFINITY; D];
        for point in self.mesh.coordinates() {
            (0..D).for_each(|axis| {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            });
        }
        let diagonal = (0..D)
            .map(|axis| (max[axis] - min[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        self.smooth_welded_with_tolerance(smoothing, WELD_TOLERANCE * diagonal);
    }
    pub(crate) fn smooth_welded_with_tolerance(&mut self, smoothing: Smoothing, tolerance: f64) {
        let mut representatives = Vec::with_capacity(self.mesh.number_of_nodes());
        let mut anchors: HashMap<[i64; D], Vec<usize>> = HashMap::new();
        let mut welded = Coordinates::new();
        for point in self.mesh.coordinates() {
            let cell = from_fn(|axis| (point[axis] / tolerance).floor() as i64);
            let mut representative = None;
            'search: for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if let Some(indices) =
                            anchors.get(&[cell[0] + dx, cell[1] + dy, cell[2] + dz])
                        {
                            for &index in indices {
                                if (point - &welded[index]).norm() <= tolerance {
                                    representative = Some(index);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
            representatives.push(representative.unwrap_or_else(|| {
                let index = welded.len();
                welded.push(point.clone());
                anchors.entry(cell).or_default().push(index);
                index
            }));
        }
        let triangles = match &self.mesh.connectivities()[0] {
            Connectivity::Triangular(triangles) => triangles
                .iter()
                .map(|&[a, b, c]| [representatives[a], representatives[b], representatives[c]])
                .collect::<Vec<_>>(),
            _ => panic!(),
        };
        let mut mesh = Mesh::from((vec![Connectivity::Triangular(triangles.into())], welded));
        mesh.smooth(smoothing);
        let smoothed = mesh.coordinates();
        self.mesh
            .coordinates
            .iter_mut()
            .zip(&representatives)
            .for_each(|(point, &representative)| *point = smoothed[representative].clone());
        self.refresh();
    }
    fn refresh(&mut self) {
        self.normals = self.mesh.normals();
        self.bvh = OnceCell::new();
    }
}
