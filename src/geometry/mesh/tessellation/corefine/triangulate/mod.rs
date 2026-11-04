//! The Delaunay triangulation of the points on and inside one triangle.
//!
//! The boundary of the triangle is kept: the points on each of its edges split it into sub-edges
//! that are never flipped. Every decision is exact, in the frame the points are given in, with
//! circles measured by a quadratic form. Where points are cocircular the choice is made as if each
//! point's height on the paraboloid were raised by a power of a tiny number that falls with its
//! rank, its index, so that the result is unique and does not depend on the order of insertion.

#[cfg(test)]
mod test;

use crate::geometry::predicates::{incircle_metric, orient2d_planar};
use std::cmp::Ordering;

/// The points to triangulate.
pub(super) struct Input {
    /// The quadratic form `[g₀, g₁, g₂]` that circles are measured with, as for `incircle_metric`.
    pub gram: [f64; 3],
    /// The points, with the three corners of the triangle first, counterclockwise. The index of a
    /// point is its rank, which decides cocircular ties.
    pub points: Vec<[f64; 2]>,
    /// The points strictly inside each edge of the triangle, from the corner the edge starts at,
    /// in order. Edge `i` runs from corner `i` to corner `i + 1`.
    pub sides: [Vec<usize>; 3],
}

impl Input {
    /// The points that are neither corners nor on an edge.
    fn interior(&self) -> Vec<usize> {
        let mut on_side = vec![false; self.points.len()];
        self.sides
            .iter()
            .flatten()
            .for_each(|&point| on_side[point] = true);
        (3..self.points.len())
            .filter(|&point| !on_side[point])
            .collect()
    }
}

/// The triangles, counterclockwise, each starting at its smallest index, in order.
pub(super) fn triangulate(input: &Input) -> Result<Vec<[usize; 3]>, &'static str> {
    Ok(build(input, [0, 1, 2], &input.interior())?.triangles())
}

/// The triangulation, inserting the points of the edges in the given order of edges, then the
/// points inside in the given order. The result does not depend on either.
fn build<'a>(
    input: &'a Input,
    edges: [usize; 3],
    interior: &[usize],
) -> Result<Triangulation<'a>, &'static str> {
    let mut triangulation = Triangulation::new(input)?;
    for edge in edges {
        for &point in &input.sides[edge] {
            triangulation.insert_on_side(edge, point)?;
        }
    }
    for &point in interior {
        triangulation.insert(point)?;
    }
    Ok(triangulation)
}

/// What is across an edge of a triangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Across {
    /// Another triangle.
    Face(usize),
    /// The outside, across the edge of the original triangle with this index.
    Side(usize),
}

/// A triangle, counterclockwise, with the neighbour opposite each of its vertices.
#[derive(Clone, Copy)]
struct Face {
    nodes: [usize; 3],
    across: [Across; 3],
}

/// Where a point is among the triangles.
enum Place {
    /// Strictly inside a triangle.
    Inside(usize),
    /// On the edge of a triangle opposite its vertex of this index.
    OnEdge(usize, usize),
}

struct Triangulation<'a> {
    input: &'a Input,
    faces: Vec<Face>,
}

impl<'a> Triangulation<'a> {
    fn new(input: &'a Input) -> Result<Self, &'static str> {
        if input.points.len() < 3 {
            return Err("a triangle needs three corners");
        }
        let corner = |index: usize| input.points[index];
        if orient2d_planar(corner(0), corner(1), corner(2)) != Ordering::Greater {
            return Err("the corners must run counterclockwise");
        }
        for (edge, points) in input.sides.iter().enumerate() {
            for &point in points {
                if point < 3
                    || orient2d_planar(corner(edge), corner((edge + 1) % 3), input.points[point])
                        != Ordering::Equal
                {
                    return Err("a point of an edge is not on the edge");
                }
            }
        }
        Ok(Self {
            input,
            faces: vec![Face {
                nodes: [0, 1, 2],
                across: [Across::Side(1), Across::Side(2), Across::Side(0)],
            }],
        })
    }
    fn point(&self, node: usize) -> [f64; 2] {
        self.input.points[node]
    }
    fn orient(&self, a: usize, b: usize, c: usize) -> Ordering {
        orient2d_planar(self.point(a), self.point(b), self.point(c))
    }
    /// Whether the edge from `b` to `c` of the counterclockwise triangle `a`, `b`, `c` must be
    /// flipped, because `d` across it is inside the circle through the triangle.
    ///
    /// On the circle the diagonal goes where it avoids the point of lowest rank of the four, as if
    /// that point were raised the most.
    fn illegal(&self, a: usize, b: usize, c: usize, d: usize) -> bool {
        let [pa, pb, pc, pd] = [a, b, c, d].map(|node| self.point(node));
        match incircle_metric(pa, pb, pc, pd, self.input.gram) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => {
                let lowest = a.min(b).min(c).min(d);
                lowest == b || lowest == c
            }
        }
    }
    /// Where a point not yet inserted is, looking through the triangles in order.
    fn locate(&self, node: usize) -> Result<Place, &'static str> {
        for (index, face) in self.faces.iter().enumerate() {
            let [a, b, c] = face.nodes;
            let signs = [
                self.orient(b, c, node),
                self.orient(c, a, node),
                self.orient(a, b, node),
            ];
            if signs.contains(&Ordering::Less) {
                continue;
            }
            let zeros: Vec<usize> = (0..3).filter(|&i| signs[i] == Ordering::Equal).collect();
            return match zeros[..] {
                [] => Ok(Place::Inside(index)),
                [edge] => Ok(Place::OnEdge(index, edge)),
                _ => Err("a point lies on another point"),
            };
        }
        Err("a point lies outside the triangle")
    }
    /// Inserts a point inside the triangle or on the edge between two triangles.
    fn insert(&mut self, node: usize) -> Result<(), &'static str> {
        let created = match self.locate(node)? {
            Place::Inside(face) => self.split_face(face, node),
            Place::OnEdge(face, edge) => {
                if matches!(self.faces[face].across[edge], Across::Side(_)) {
                    return Err("a point inside lies on the boundary");
                }
                self.split_edge(face, edge, node)
            }
        };
        self.legalize(node, created);
        Ok(())
    }
    /// Inserts a point on an edge of the original triangle, which must come after the points of
    /// that edge before it, since it splits the last piece of the edge.
    fn insert_on_side(&mut self, side: usize, node: usize) -> Result<(), &'static str> {
        let end = (side + 1) % 3;
        let (face, edge) = (0..self.faces.len())
            .find_map(|face| {
                (0..3)
                    .find(|&edge| {
                        self.faces[face].across[edge] == Across::Side(side)
                            && self.faces[face].nodes[(edge + 2) % 3] == end
                    })
                    .map(|edge| (face, edge))
            })
            .ok_or("the edge of the triangle has no piece left to split")?;
        let created = self.split_edge(face, edge, node);
        self.legalize(node, created);
        Ok(())
    }
    /// Points the neighbour across `edge`, as it appears in the neighbour, at the triangle `to`.
    fn retarget(&mut self, across: Across, edge: [usize; 2], to: usize) {
        if let Across::Face(neighbour) = across {
            let face = &mut self.faces[neighbour];
            let slot = (0..3)
                .find(|&k| face.nodes[(k + 1) % 3] == edge[0] && face.nodes[(k + 2) % 3] == edge[1])
                .expect("the neighbour does not share the edge");
            face.across[slot] = Across::Face(to);
        }
    }
    /// Splits a triangle into three around a point inside it, and returns the new triangles.
    fn split_face(&mut self, f: usize, p: usize) -> Vec<usize> {
        let Face {
            nodes: [a, b, c],
            across: [x_bc, x_ca, x_ab],
        } = self.faces[f];
        let (n1, n2) = (self.faces.len(), self.faces.len() + 1);
        self.faces[f] = Face {
            nodes: [a, b, p],
            across: [Across::Face(n1), Across::Face(n2), x_ab],
        };
        self.faces.push(Face {
            nodes: [b, c, p],
            across: [Across::Face(n2), Across::Face(f), x_bc],
        });
        self.faces.push(Face {
            nodes: [c, a, p],
            across: [Across::Face(f), Across::Face(n1), x_ca],
        });
        self.retarget(x_bc, [c, b], n1);
        self.retarget(x_ca, [a, c], n2);
        vec![f, n1, n2]
    }
    /// Splits the edge of a triangle opposite its vertex `i` at a point on it, and the triangle
    /// across it if there is one, and returns the new triangles.
    fn split_edge(&mut self, f: usize, i: usize, p: usize) -> Vec<usize> {
        let Face { nodes, across } = self.faces[f];
        let (a, u, v) = (nodes[i], nodes[(i + 1) % 3], nodes[(i + 2) % 3]);
        let (x_uv, x_va, x_au) = (across[i], across[(i + 1) % 3], across[(i + 2) % 3]);
        let f2 = self.faces.len();
        let Across::Face(g) = x_uv else {
            self.faces[f] = Face {
                nodes: [a, u, p],
                across: [x_uv, Across::Face(f2), x_au],
            };
            self.faces.push(Face {
                nodes: [a, p, v],
                across: [x_uv, x_va, Across::Face(f)],
            });
            self.retarget(x_va, [a, v], f2);
            return vec![f, f2];
        };
        let j = self.faces[g]
            .across
            .iter()
            .position(|&x| x == Across::Face(f))
            .expect("the neighbour does not point back");
        let d = self.faces[g].nodes[j];
        let (y_ud, y_dv) = (
            self.faces[g].across[(j + 1) % 3],
            self.faces[g].across[(j + 2) % 3],
        );
        let g2 = f2 + 1;
        self.faces[f] = Face {
            nodes: [a, u, p],
            across: [Across::Face(g2), Across::Face(f2), x_au],
        };
        self.faces.push(Face {
            nodes: [a, p, v],
            across: [Across::Face(g), x_va, Across::Face(f)],
        });
        self.faces[g] = Face {
            nodes: [d, v, p],
            across: [Across::Face(f2), Across::Face(g2), y_dv],
        };
        self.faces.push(Face {
            nodes: [d, p, u],
            across: [Across::Face(f), y_ud, Across::Face(g)],
        });
        self.retarget(x_va, [a, v], f2);
        self.retarget(y_ud, [d, u], g2);
        vec![f, f2, g, g2]
    }
    /// Flips every edge opposite a new point whose triangle across it has a point inside its
    /// circle, and the edges that exposes, until none does.
    fn legalize(&mut self, p: usize, mut stack: Vec<usize>) {
        while let Some(f) = stack.pop() {
            let i = self.faces[f]
                .nodes
                .iter()
                .position(|&node| node == p)
                .expect("the triangle does not have the point");
            let Across::Face(g) = self.faces[f].across[i] else {
                continue;
            };
            let [a, b, c] = [0, 1, 2].map(|k| self.faces[f].nodes[(i + k) % 3]);
            let j = self.faces[g]
                .across
                .iter()
                .position(|&x| x == Across::Face(f))
                .expect("the neighbour does not point back");
            let d = self.faces[g].nodes[j];
            if self.illegal(a, b, c, d) {
                self.flip(f, i, g, j);
                stack.push(f);
                stack.push(g);
            }
        }
    }
    /// Flips the edge between triangle `f`, whose vertex `i` is opposite it, and triangle `g`,
    /// whose vertex `j` is. The new triangles keep the indices and the vertex `a` of `f`.
    fn flip(&mut self, f: usize, i: usize, g: usize, j: usize) {
        let (face, other) = (self.faces[f], self.faces[g]);
        let [a, b, c] = [0, 1, 2].map(|k| face.nodes[(i + k) % 3]);
        let (x_ca, x_ab) = (face.across[(i + 1) % 3], face.across[(i + 2) % 3]);
        let d = other.nodes[j];
        let (y_bd, y_dc) = (other.across[(j + 1) % 3], other.across[(j + 2) % 3]);
        self.faces[f] = Face {
            nodes: [a, b, d],
            across: [y_bd, Across::Face(g), x_ab],
        };
        self.faces[g] = Face {
            nodes: [a, d, c],
            across: [y_dc, x_ca, Across::Face(f)],
        };
        self.retarget(y_bd, [d, b], f);
        self.retarget(x_ca, [a, c], g);
    }
    /// The triangles, each starting at its smallest index, in order.
    fn triangles(&self) -> Vec<[usize; 3]> {
        let mut triangles: Vec<[usize; 3]> = self
            .faces
            .iter()
            .map(|face| {
                let start = (0..3).min_by_key(|&k| face.nodes[k]).unwrap_or(0);
                [0, 1, 2].map(|k| face.nodes[(start + k) % 3])
            })
            .collect();
        triangles.sort_unstable();
        triangles
    }
}
