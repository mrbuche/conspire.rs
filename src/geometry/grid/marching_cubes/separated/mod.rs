#[cfg(test)]
pub(crate) mod test;

use super::{Gradient, Isosurface};
use crate::{
    geometry::{Coordinate, Coordinates, Direction, Directions, grid::Voxels},
    math::{FxHashMap, TensorVec},
};
use std::array::from_fn;

pub(crate) type Corner = [usize; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Vertex {
    Inside(Corner),
    Boundary([Corner; 2]),
}

impl Vertex {
    pub(crate) fn boundary(one: Corner, two: Corner) -> Self {
        if one < two {
            Self::Boundary([one, two])
        } else {
            Self::Boundary([two, one])
        }
    }
}

pub(crate) const CORNERS: [Corner; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

pub(crate) const FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

pub(crate) struct Polyhedron {
    pub(crate) faces: Vec<Vec<Vertex>>,
}

impl Polyhedron {
    pub(crate) fn vertices(&self) -> Vec<Vertex> {
        let mut vertices: Vec<Vertex> = self.faces.iter().flatten().copied().collect();
        vertices.sort_unstable();
        vertices.dedup();
        vertices
    }
    /// The faces lying in the surface rather than in a face of the cell.
    ///
    /// A face of the cell that is clipped keeps at least one corner of the
    /// cell, whereas a cut runs between points on edges alone.
    pub(crate) fn cuts(&self) -> impl Iterator<Item = &Vec<Vertex>> {
        self.faces.iter().filter(|face| {
            face.iter()
                .all(|vertex| matches!(vertex, Vertex::Boundary(_)))
        })
    }
}

/// Whether a sample lies within the object.
pub(crate) fn inside(value: f64, level: f64, gradient: Gradient) -> bool {
    match gradient {
        Gradient::Descent => value >= level,
        Gradient::Ascent => value <= level,
    }
}

fn clip(corners: [Corner; 4], inside: [bool; 4]) -> (Vec<Vec<Vertex>>, Vec<[Vertex; 2]>) {
    if inside.iter().all(|&flag| flag) {
        return (vec![corners.map(Vertex::Inside).to_vec()], Vec::new());
    } else if inside.iter().all(|&flag| !flag) {
        return (Vec::new(), Vec::new());
    }
    if inside[0] == inside[2] && inside[1] == inside[3] {
        let (mut polygons, mut cuts) = (Vec::new(), Vec::new());
        (0..4).filter(|&i| inside[i]).for_each(|i| {
            let before = Vertex::boundary(corners[(i + 3) % 4], corners[i]);
            let after = Vertex::boundary(corners[i], corners[(i + 1) % 4]);
            polygons.push(vec![before, Vertex::Inside(corners[i]), after]);
            cuts.push([after, before])
        });
        return (polygons, cuts);
    }
    let mut walk = Vec::new();
    (0..4).for_each(|i| {
        let next = (i + 1) % 4;
        if inside[i] {
            walk.push(Vertex::Inside(corners[i]))
        }
        if inside[i] != inside[next] {
            walk.push(Vertex::boundary(corners[i], corners[next]))
        }
    });
    let cuts = (0..walk.len())
        .filter_map(|i| {
            let next = (i + 1) % walk.len();
            matches!(
                (walk[i], walk[next]),
                (Vertex::Boundary(_), Vertex::Boundary(_))
            )
            .then_some([walk[i], walk[next]])
        })
        .collect();
    (vec![walk], cuts)
}

fn loops(cuts: Vec<[Vertex; 2]>) -> Result<Vec<Vec<Vertex>>, &'static str> {
    let mut next = FxHashMap::default();
    for [from, to] in cuts {
        if next.insert(to, from).is_some() {
            return Err("a cell is cut more than once along the same edge");
        }
    }
    let mut chains = Vec::new();
    while let Some(&start) = next.keys().min() {
        let mut chain = vec![start];
        let mut here = next.remove(&start).ok_or("open cut chain within a cell")?;
        while here != start {
            chain.push(here);
            here = next.remove(&here).ok_or("open cut chain within a cell")?;
        }
        if chain.len() < 3 {
            return Err("a cut leaves a degenerate face");
        }
        chains.push(chain)
    }
    Ok(chains)
}

fn pieces(mut faces: Vec<Vec<Vertex>>) -> Vec<Polyhedron> {
    let mut pieces = Vec::new();
    while let Some(first) = faces.pop() {
        let mut vertices: Vec<Vertex> = first.clone();
        let mut piece = vec![first];
        let mut grown = true;
        while grown {
            grown = false;
            faces.retain(|face| {
                if face.iter().any(|vertex| vertices.contains(vertex)) {
                    vertices.extend(face.iter().copied());
                    piece.push(face.clone());
                    grown = true;
                    false
                } else {
                    true
                }
            })
        }
        pieces.push(Polyhedron { faces: piece })
    }
    pieces
}

/// Clips a cell to the object, giving the polyhedra that are left.
///
/// The topology follows from the signs at the corners alone, and every face
/// of the cell whose signs alternate is cut so as to keep them apart. That is
/// the only choice two cells sharing such a face are bound to make alike.
pub(crate) fn cell(
    corners: [Corner; 8],
    inside: [bool; 8],
) -> Result<Vec<Polyhedron>, &'static str> {
    let mut faces = Vec::new();
    let mut cuts = Vec::new();
    for face in FACES {
        let (polygons, edges) = clip(
            face.map(|local| corners[local]),
            face.map(|local| inside[local]),
        );
        faces.extend(polygons);
        cuts.extend(edges)
    }
    loops(cuts).map(|chains| {
        faces.extend(chains);
        pieces(faces)
    })
}

/// Extracts the isosurface with the topology of [`cell`], each cut a fan of
/// triangles, so that it is the boundary of the hexahedra clipped from the
/// same samples. The triangles wind as those of the other methods do, which
/// is against the hand of the samples' axes.
pub(super) fn extract(
    volume: &Voxels<f64>,
    mask: Option<&Voxels<bool>>,
    level: f64,
    gradient: Gradient,
    spacing: [f64; 3],
) -> Result<Isosurface, &'static str> {
    let nel = *volume.nel();
    let data = volume.data();
    let value = |corner: Corner| data[volume.flat(corner)];
    let slope = |corner: Corner| -> [f64; 3] {
        from_fn(|axis| {
            let low = corner[axis].saturating_sub(1);
            let high = (corner[axis] + 1).min(nel[axis] - 1);
            let (mut below, mut above) = (corner, corner);
            below[axis] = low;
            above[axis] = high;
            (value(above) - value(below)) / ((high - low) as f64 * spacing[axis])
        })
    };
    let outward = match gradient {
        Gradient::Descent => -1.0,
        Gradient::Ascent => 1.0,
    };
    let mut indices = FxHashMap::<[Corner; 2], usize>::default();
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut normals: Vec<[f64; 3]> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    for k in 0..nel[2] - 1 {
        for j in 0..nel[1] - 1 {
            for i in 0..nel[0] - 1 {
                if let Some(mask) = mask
                    && !mask.data()[mask.flat([i + 1, j + 1, k + 1])]
                {
                    continue;
                }
                let corners = CORNERS.map(|[a, b, c]| [i + a, j + b, k + c]);
                let flags = corners.map(|corner| inside(value(corner), level, gradient));
                if flags.iter().all(|&flag| flag) || flags.iter().all(|&flag| !flag) {
                    continue;
                }
                for polyhedron in cell(corners, flags)? {
                    for cut in polyhedron.cuts() {
                        let ids: Vec<usize> = cut
                            .iter()
                            .map(|vertex| {
                                let Vertex::Boundary(edge) = *vertex else {
                                    unreachable!("a cut runs between edges")
                                };
                                *indices.entry(edge).or_insert_with(|| {
                                    let [one, two] = edge;
                                    let (a, b) = (value(one), value(two));
                                    let t = (level - a) / (b - a);
                                    vertices.push(from_fn(|axis| {
                                        let (p, q) = (one[axis] as f64, two[axis] as f64);
                                        (p + t * (q - p)) * spacing[axis]
                                    }));
                                    let (ga, gb) = (slope(one), slope(two));
                                    let direction: [f64; 3] =
                                        from_fn(|axis| ga[axis] + t * (gb[axis] - ga[axis]));
                                    let length = direction
                                        .iter()
                                        .map(|component| component * component)
                                        .sum::<f64>()
                                        .sqrt();
                                    normals.push(if length > 0.0 {
                                        direction.map(|component| outward * component / length)
                                    } else {
                                        [0.0; 3]
                                    });
                                    values.push(a.max(b));
                                    vertices.len() - 1
                                })
                            })
                            .collect();
                        (1..ids.len() - 1)
                            .for_each(|index| faces.push([ids[0], ids[index + 1], ids[index]]))
                    }
                }
            }
        }
    }
    if vertices.is_empty() {
        return Err("No surface found at the given iso value.");
    }
    let mut coordinates = Coordinates::new();
    vertices
        .into_iter()
        .for_each(|vertex| coordinates.push(Coordinate::const_from(vertex)));
    let mut directions = Directions::new();
    normals
        .into_iter()
        .for_each(|normal| directions.push(Direction::const_from(normal)));
    Ok(Isosurface {
        vertices: coordinates,
        faces,
        normals: directions,
        values,
    })
}
