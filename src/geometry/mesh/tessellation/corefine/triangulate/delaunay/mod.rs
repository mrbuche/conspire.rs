//! Restoring the Delaunay property by flipping edges.

use super::{Across, Face, Triangulation};
use crate::geometry::predicates::incircle_metric;
use core::cmp::Ordering;

impl Triangulation<'_> {
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
    /// Flips every edge opposite a new point whose triangle across it has a point inside its
    /// circle, and the edges that exposes, until none does.
    pub(super) fn legalize(&mut self, p: usize, mut stack: Vec<usize>) {
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
    pub(super) fn flip(&mut self, f: usize, i: usize, g: usize, j: usize) {
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
    /// Flips edges that are not fixed until every one has the Delaunay property.
    pub(super) fn legalize_all(&mut self) -> Result<(), &'static str> {
        let size = self.input.points.len();
        let mut budget = 10 * size * size + 100;
        loop {
            let mut flipped = false;
            for f in 0..self.faces.len() {
                for i in 0..3 {
                    let Across::Face(g) = self.faces[f].across[i] else {
                        continue;
                    };
                    let [a, b, c] = [0, 1, 2].map(|k| self.faces[f].nodes[(i + k) % 3]);
                    if self.fixed.contains(&[b.min(c), b.max(c)]) {
                        continue;
                    }
                    let j = self.faces[g]
                        .across
                        .iter()
                        .position(|&x| x == Across::Face(f))
                        .expect("the neighbour does not point back");
                    let d = self.faces[g].nodes[j];
                    if self.illegal(a, b, c, d) {
                        budget = budget
                            .checked_sub(1)
                            .ok_or("restoring the Delaunay property did not finish")?;
                        self.flip(f, i, g, j);
                        flipped = true;
                    }
                }
            }
            if !flipped {
                return Ok(());
            }
        }
    }
}
