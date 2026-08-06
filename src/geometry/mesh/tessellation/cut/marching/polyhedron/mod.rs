use super::{CORNERS, Corner, FACES, Signs, Vertex};
use crate::{
    geometry::{
        Coordinate, DirectionsRef,
        mesh::tessellation::{D, Tessellation},
    },
    math::{FxHashMap, Tensor},
};

pub(super) struct Polyhedron {
    pub(super) faces: Vec<Vec<Vertex>>,
}

fn clip(
    corners: [Corner; 4],
    inside: [bool; 4],
    joined: impl FnOnce() -> bool,
) -> (Vec<Vec<Vertex>>, Vec<[Vertex; 2]>) {
    if inside.iter().all(|&flag| flag) {
        return (vec![corners.map(Vertex::Inside).to_vec()], Vec::new());
    } else if inside.iter().all(|&flag| !flag) {
        return (Vec::new(), Vec::new());
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
    let opposed = inside[0] == inside[2] && inside[1] == inside[3] && inside[0] != inside[1];
    if opposed && !joined() {
        let (mut polygons, mut cuts) = (Vec::new(), Vec::new());
        (0..4).filter(|&i| inside[i]).for_each(|i| {
            let before = Vertex::boundary(corners[(i + 3) % 4], corners[i]);
            let after = Vertex::boundary(corners[i], corners[(i + 1) % 4]);
            polygons.push(vec![before, Vertex::Inside(corners[i]), after]);
            cuts.push([after, before])
        });
        return (polygons, cuts);
    }
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

impl Tessellation {
    pub(super) fn polyhedra(
        &self,
        lattice: &super::Lattice,
        signs: &Signs,
    ) -> Result<Vec<Polyhedron>, &'static str> {
        let surface = self.mesh();
        let coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: DirectionsRef<'_, D> = self.normals().iter().flatten().collect();
        let directions = super::DIRECTIONS.map(|direction| direction.normalized());
        lattice
            .cells()
            .iter()
            .filter_map(|&([i, j, k], _)| {
                let corners = CORNERS.map(|[a, b, c]| [i + a, j + b, k + c]);
                let inside = corners.map(|corner| signs.at(corner));
                if inside.iter().all(|&flag| !flag) {
                    return None;
                }
                let mut faces = Vec::new();
                let mut cuts = Vec::new();
                for face in FACES {
                    let quad = face.map(|local| corners[local]);
                    let flags = face.map(|local| inside[local]);
                    let middle = || {
                        let point = quad
                            .iter()
                            .map(|&corner| signs.point(corner))
                            .sum::<Coordinate<D>>()
                            / 4.0;
                        self.encloses(&point, coordinates, &elements, &normals, &directions)
                    };
                    let (polygons, edges) = clip(quad, flags, middle);
                    faces.extend(polygons);
                    cuts.extend(edges)
                }
                Some(loops(cuts).map(|chains| {
                    faces.extend(chains);
                    Polyhedron { faces }
                }))
            })
            .collect()
    }
}

impl Polyhedron {
    pub(super) fn vertices(&self) -> Vec<Vertex> {
        let mut vertices: Vec<Vertex> = self.faces.iter().flatten().copied().collect();
        vertices.sort_unstable();
        vertices.dedup();
        vertices
    }
}
