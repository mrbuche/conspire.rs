//! Inserting a point into the triangulation, inside a triangle or on an edge.

use super::{Across, Face, Triangulation};
use core::cmp::Ordering;

/// Where a point is among the triangles.
enum Place {
    /// Strictly inside a triangle.
    Inside(usize),
    /// On the edge of a triangle opposite its vertex of this index.
    OnEdge(usize, usize),
}

impl Triangulation<'_> {
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
    pub(super) fn insert(&mut self, node: usize) -> Result<(), &'static str> {
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
    pub(super) fn insert_on_side(&mut self, side: usize, node: usize) -> Result<(), &'static str> {
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
    pub(super) fn retarget(&mut self, across: Across, edge: [usize; 2], to: usize) {
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
}
