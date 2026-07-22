#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
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
    collections::{HashMap, hash_map::Entry},
    thread::{available_parallelism, scope},
};

const STITCH_TRIM_MARGIN: Scalar = 1.0;
const CURVE_SEGMENTS: usize = 16;
const CURVE_RELAXATION: usize = 100;
const CORNER_ANCHOR: Scalar = 0.25;

pub struct ProjectedNetwork {
    pub core: Mesh<D>,
    pub surface: Mesh<D>,
    pub edges: Vec<[usize; 2]>,
    pub curves: Vec<Vec<Coordinate<D>>>,
}

impl Tessellation {
    pub fn fitted_core_and_surface(
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
        let quads = core.exterior_faces();
        let sizing = QuadSizing::new(&core, &quads)?;
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
        let surface = Mesh::from((vec![connectivity.into()], coordinates));
        Ok((core, surface))
    }
    pub fn projected_network(
        &self,
        balancing: Balancing,
        scale: Scalar,
        curvature: CurvatureSizing,
        iterations: usize,
    ) -> Result<ProjectedNetwork, &'static str> {
        let (core, surface) =
            self.fitted_core_and_surface(balancing, scale, curvature, iterations)?;
        let quads = core.exterior_faces();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let surface_coordinates = surface.coordinates();
        let bvh = BoundingVolumeHierarchy::from(&surface);
        let core_coordinates = core.coordinates();
        let mut edges: Vec<[usize; 2]> = quads
            .iter()
            .flat_map(|quad| {
                (0..4).map(|i| {
                    let mut edge = [quad[i], quad[(i + 1) % 4]];
                    edge.sort_unstable();
                    edge
                })
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        let project = |point: &Coordinate<D>| -> Result<Coordinate<D>, &'static str> {
            Ok(bvh
                .closest_point(point, surface_coordinates, &elements)
                .ok_or("empty surface")?
                .0)
        };
        let mut corners: HashMap<usize, Coordinate<D>> = HashMap::new();
        edges.iter().flatten().try_for_each(|&vertex| {
            if let Entry::Vacant(slot) = corners.entry(vertex) {
                slot.insert(project(&core_coordinates[vertex])?);
            }
            Ok::<(), &'static str>(())
        })?;
        let mut curves = edges
            .iter()
            .map(|&[a, b]| {
                let mut curve = vec![corners[&a].clone()];
                (1..CURVE_SEGMENTS).try_for_each(|i| {
                    let t = i as Scalar / CURVE_SEGMENTS as Scalar;
                    let sample =
                        &core_coordinates[a] + &((&core_coordinates[b] - &core_coordinates[a]) * t);
                    curve.push(project(&sample)?);
                    Ok::<(), &'static str>(())
                })?;
                curve.push(corners[&b].clone());
                Ok(curve)
            })
            .collect::<Result<Vec<Vec<Coordinate<D>>>, &'static str>>()?;
        let mut incident: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();
        edges.iter().enumerate().for_each(|(index, &[a, b])| {
            incident.entry(a).or_default().push((index, true));
            incident.entry(b).or_default().push((index, false));
        });
        let mut vertices: Vec<usize> = incident.keys().copied().collect();
        vertices.sort_unstable();
        (0..CURVE_RELAXATION).try_for_each(|_| {
            curves.iter_mut().try_for_each(|curve| {
                (1..curve.len() - 1).try_for_each(|i| {
                    let midpoint = (&curve[i - 1] + &curve[i + 1]) / 2.0;
                    curve[i] = project(&midpoint)?;
                    Ok::<(), &'static str>(())
                })
            })?;
            vertices.iter().try_for_each(|vertex| {
                let list = &incident[vertex];
                let mean = list
                    .iter()
                    .map(|&(edge, at_start)| {
                        let curve = &curves[edge];
                        if at_start {
                            curve[1].clone()
                        } else {
                            curve[curve.len() - 2].clone()
                        }
                    })
                    .sum::<Coordinate<D>>()
                    / list.len() as Scalar;
                let blend = &(mean * (1.0 - CORNER_ANCHOR)) + &(&corners[vertex] * CORNER_ANCHOR);
                let point = project(&blend)?;
                list.iter().for_each(|&(edge, at_start)| {
                    let curve = &mut curves[edge];
                    if at_start {
                        curve[0] = point.clone();
                    } else {
                        let last = curve.len() - 1;
                        curve[last] = point.clone();
                    }
                });
                Ok::<(), &'static str>(())
            })
        })?;
        Ok(ProjectedNetwork {
            core,
            surface,
            edges,
            curves,
        })
    }
}

struct QuadSizing {
    bvh: BoundingVolumeHierarchy<D>,
    coordinates: Coordinates<D>,
    triangles: Vec<[usize; 3]>,
    lengths: Vec<Scalar>,
}

impl QuadSizing {
    fn new(core: &Mesh<D>, quads: &[Vec<usize>]) -> Result<Self, &'static str> {
        let coordinates = core.coordinates();
        if quads.iter().any(|quad| quad.len() != 4) {
            return Err("fitted_core_and_surface requires an all-hexahedral trimmed core");
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
    fn nearest_quads(&self, points: &Coordinates<D>) -> Vec<Option<usize>> {
        let elements: Vec<&[usize]> = self.triangles.iter().map(|t| t.as_slice()).collect();
        let number_of_points = points.len();
        let mut nearest = vec![None; number_of_points];
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = number_of_points.div_ceil(threads).max(1);
        scope(|scope| {
            let (bvh, coordinates, elements) = (&self.bvh, &self.coordinates, &elements);
            nearest
                .chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk, out)| {
                    scope.spawn(move || {
                        let offset = chunk * chunk_size;
                        out.iter_mut().enumerate().for_each(|(local, slot)| {
                            let point = &points[offset + local];
                            *slot = bvh
                                .closest_point(point, coordinates, elements)
                                .map(|(_, triangle)| triangle / 2);
                        });
                    });
                });
        });
        nearest
    }
    fn target_lengths(&self, points: &Coordinates<D>) -> Vec<Scalar> {
        self.nearest_quads(points)
            .into_iter()
            .map(|quad| quad.map_or(Scalar::INFINITY, |quad| self.lengths[quad]))
            .collect()
    }
}
