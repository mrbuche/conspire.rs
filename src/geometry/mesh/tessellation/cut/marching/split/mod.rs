use super::{Signs, Vertex, polyhedron::Polyhedron};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Mesh,
            tessellation::{D, Tessellation},
        },
    },
    math::{FxHashMap, Tensor, TensorVec},
};

/// A node of the split mesh, named by what it is the middle of so that cells
/// agree on the ones they share without comparing coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Node {
    Vertex(Vertex),
    Edge([Vertex; 2]),
    Face(Vec<Vertex>),
    Body(usize),
}

/// The edge a face leaves a vertex by, the one it arrives on, and the face.
type Around = ([Vertex; 2], [Vertex; 2], Vec<Vertex>);

fn edge(one: Vertex, two: Vertex) -> [Vertex; 2] {
    if one < two { [one, two] } else { [two, one] }
}

fn key(face: &[Vertex]) -> Vec<Vertex> {
    let mut key = face.to_vec();
    key.sort_unstable();
    key
}

/// Splits each polyhedron about the middles of its edges, faces and body,
/// which leaves one hexahedron per vertex and matches across cells because
/// every node is named by the vertices it lies among.
pub(super) fn hexahedra(
    cells: Vec<Polyhedron>,
    points: &FxHashMap<Vertex, Coordinate<D>>,
) -> Result<Mesh<D>, &'static str> {
    let mut connectivity = Vec::new();
    let mut positions = FxHashMap::default();
    for (cell, polyhedron) in cells.iter().enumerate() {
        let vertices = polyhedron.vertices();
        let body = vertices
            .iter()
            .map(|vertex| points[vertex].clone())
            .sum::<Coordinate<D>>()
            / vertices.len() as f64;
        positions.insert(Node::Body(cell), body);
        let mut links: FxHashMap<Vertex, Vec<Around>> = FxHashMap::default();
        for face in polyhedron.faces.iter() {
            let middle = face
                .iter()
                .map(|vertex| points[vertex].clone())
                .sum::<Coordinate<D>>()
                / face.len() as f64;
            positions.insert(Node::Face(key(face)), middle);
            for (index, &vertex) in face.iter().enumerate() {
                let previous = face[(index + face.len() - 1) % face.len()];
                let next = face[(index + 1) % face.len()];
                links.entry(vertex).or_default().push((
                    edge(vertex, next),
                    edge(vertex, previous),
                    key(face),
                ))
            }
        }
        for &vertex in vertices.iter() {
            let around = &links[&vertex];
            if around.len() != 3 {
                return Err("a cut leaves a cell that is not three-regular");
            }
            let mut cycle = Vec::new();
            let mut here = around[0].0;
            for _ in 0..3 {
                let (_, previous, face) = around
                    .iter()
                    .find(|(from, _, _)| *from == here)
                    .ok_or("a cut leaves a cell whose faces do not close about a vertex")?;
                cycle.push((here, face.clone()));
                here = *previous
            }
            if here != around[0].0 {
                return Err("a cut leaves a cell whose faces do not close about a vertex");
            }
            let [(one, first), (two, second), (three, third)] =
                <[_; 3]>::try_from(cycle).map_err(|_| "unreachable")?;
            [one, two, three].iter().for_each(|pair| {
                positions
                    .entry(Node::Edge(*pair))
                    .or_insert_with(|| (&points[&pair[0]] + &points[&pair[1]]) / 2.0);
            });
            connectivity.push([
                Node::Vertex(vertex),
                Node::Edge(two),
                Node::Face(first),
                Node::Edge(one),
                Node::Edge(three),
                Node::Face(second),
                Node::Body(cell),
                Node::Face(third),
            ])
        }
        vertices.iter().for_each(|&vertex| {
            positions.insert(Node::Vertex(vertex), points[&vertex].clone());
        })
    }
    let mut nodes: Vec<Node> = positions.keys().cloned().collect();
    nodes.sort_unstable();
    let numbering: FxHashMap<&Node, usize> = nodes
        .iter()
        .enumerate()
        .map(|(number, node)| (node, number))
        .collect();
    let mut coordinates = Coordinates::new();
    nodes
        .iter()
        .for_each(|node| coordinates.push(positions[node].clone()));
    let hexes: Vec<[usize; 8]> = connectivity
        .iter()
        .map(|hex| std::array::from_fn(|corner| numbering[&hex[corner]]))
        .collect();
    Ok(Mesh::from((
        vec![Connectivity::Hexahedral(hexes.into())],
        coordinates,
    )))
}

impl Tessellation {
    /// Places a vertex on every edge whose ends straddle the surface.
    pub(super) fn placements(
        &self,
        cells: &[Polyhedron],
        signs: &Signs,
        placement: super::Placement,
    ) -> Result<FxHashMap<Vertex, Coordinate<D>>, &'static str> {
        let surface = self.mesh();
        let coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = self.bvh();
        let mut points = FxHashMap::default();
        for polyhedron in cells {
            for vertex in polyhedron.vertices() {
                if points.contains_key(&vertex) {
                    continue;
                }
                let point = match vertex {
                    Vertex::Inside(corner) => signs.point(corner),
                    Vertex::Boundary([one, two]) => {
                        let (start, end) = (signs.point(one), signs.point(two));
                        let fraction = match placement {
                            super::Placement::Midpoint => 0.5,
                            super::Placement::Crossing(guard) => {
                                let along = &end - &start;
                                let length = along.norm();
                                let ray = (start.clone(), along / length).into();
                                match bvh.intersect(&ray, coordinates, &elements) {
                                    Some(hit) => {
                                        (hit.distance() / length).clamp(guard, 1.0 - guard)
                                    }
                                    None => 0.5,
                                }
                            }
                        };
                        &start + &((&end - &start) * fraction)
                    }
                };
                points.insert(vertex, point);
            }
        }
        Ok(points)
    }
}
