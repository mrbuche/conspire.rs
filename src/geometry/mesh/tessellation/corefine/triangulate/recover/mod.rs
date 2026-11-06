//! Making the segments that are asked for edges of the triangulation.[^1]
//!
//! [^1]: S. W. Sloan, [Comput. Struct. **47**, 441 (1993)](https://doi.org/10.1016/0045-7949(93)90239-A).

use super::{Across, Triangulation};
use alloc::collections::VecDeque;
use core::cmp::Ordering;

/// How far a segment gets before it meets a point, or the edges it crosses.
enum Walk {
    /// The segment passes through this point.
    Through(usize),
    /// The segment crosses these edges, as pairs of the point to its left and to its right, in
    /// order from its start.
    Crossed(Vec<[usize; 2]>),
}

impl Triangulation<'_> {
    /// Whether two points are joined by an edge.
    fn has_edge(&self, u: usize, v: usize) -> bool {
        self.faces
            .iter()
            .any(|face| face.nodes.contains(&u) && face.nodes.contains(&v))
    }
    /// Makes every segment an edge that cannot flip, joining through any point it passes through.
    pub(super) fn recover(&mut self, segments: &[[usize; 2]]) -> Result<(), &'static str> {
        let mut work: Vec<[usize; 2]> = segments.iter().rev().copied().collect();
        while let Some([u, v]) = work.pop() {
            if u == v {
                return Err("a segment has no length");
            }
            if let Some(through) = self.recover_one(u, v)? {
                work.push([through, v]);
                work.push([u, through]);
            }
        }
        Ok(())
    }
    /// Makes a segment an edge, or says which point it passes through, in which case nothing has
    /// changed.
    fn recover_one(&mut self, u: usize, v: usize) -> Result<Option<usize>, &'static str> {
        if self.has_edge(u, v) {
            self.fixed.insert([u.min(v), u.max(v)]);
            return Ok(None);
        }
        let crossed = match self.walk(u, v)? {
            Walk::Through(point) => return Ok(Some(point)),
            Walk::Crossed(crossed) => crossed,
        };
        let size = self.input.points.len();
        let mut budget = 10 * size * size + 100;
        let mut queue: VecDeque<[usize; 2]> = crossed.into();
        while let Some([p, q]) = queue.pop_front() {
            budget = budget
                .checked_sub(1)
                .ok_or("recovering a segment did not finish")?;
            if self.fixed.contains(&[p.min(q), p.max(q)]) {
                return Err("two segments cross");
            }
            let (f, i, g, j) = self
                .faces_on(p, q)
                .ok_or("an edge crossing a segment is lost")?;
            let (x, y) = (self.faces[f].nodes[i], self.faces[g].nodes[j]);
            let (side_p, side_q) = (self.orient(x, y, p), self.orient(x, y, q));
            if side_p == Ordering::Equal || side_q == Ordering::Equal || side_p == side_q {
                queue.push_back([p, q]);
                continue;
            }
            self.flip(f, i, g, j);
            let (side_x, side_y) = (self.orient(u, v, x), self.orient(u, v, y));
            if side_x != Ordering::Equal && side_y != Ordering::Equal && side_x != side_y {
                queue.push_back([x, y]);
            }
        }
        if !self.has_edge(u, v) {
            return Err("a segment was not recovered");
        }
        self.fixed.insert([u.min(v), u.max(v)]);
        Ok(None)
    }
    /// The edges a segment crosses, or the first point it passes through on the way.
    fn walk(&self, u: usize, v: usize) -> Result<Walk, &'static str> {
        let (mut left, mut right, mut face) = 'wedge: {
            for (f, face) in self.faces.iter().enumerate() {
                let Some(i) = face.nodes.iter().position(|&node| node == u) else {
                    continue;
                };
                let (b, c) = (face.nodes[(i + 1) % 3], face.nodes[(i + 2) % 3]);
                for point in [b, c] {
                    if self.orient(u, point, v) == Ordering::Equal && self.beyond(u, point, v) {
                        return self.through(u, point, v);
                    }
                }
                if self.orient(u, b, v) == Ordering::Greater
                    && self.orient(u, c, v) == Ordering::Less
                {
                    break 'wedge (c, b, f);
                }
            }
            return Err("a segment leaves the triangle");
        };
        let mut crossed = vec![[left, right]];
        loop {
            let slot = (0..3)
                .find(|&k| ![left, right].contains(&self.faces[face].nodes[k]))
                .expect("the triangle has no vertex off the edge");
            let Across::Face(next) = self.faces[face].across[slot] else {
                return Err("a segment leaves the triangle");
            };
            let apex = (0..3)
                .map(|k| self.faces[next].nodes[k])
                .find(|node| ![left, right].contains(node))
                .expect("the neighbour has no vertex off the edge");
            if apex == v {
                return Ok(Walk::Crossed(crossed));
            }
            match self.orient(u, v, apex) {
                Ordering::Equal => return self.through(u, apex, v),
                Ordering::Greater => left = apex,
                Ordering::Less => right = apex,
            }
            crossed.push([left, right]);
            face = next;
        }
    }
    /// Whether `point`, collinear with `u` and `v`, is on the same side of `u` as `v`.
    fn beyond(&self, u: usize, point: usize, v: usize) -> bool {
        let (u, point, v) = (self.point(u), self.point(point), self.point(v));
        let axis = if u[0] != v[0] { 0 } else { 1 };
        point[axis] != u[axis] && (point[axis] > u[axis]) == (v[axis] > u[axis])
    }
    /// The point a segment passes through, which has to be between its ends.
    fn through(&self, u: usize, point: usize, v: usize) -> Result<Walk, &'static str> {
        let (a, b, c) = (self.point(u), self.point(point), self.point(v));
        let axis = if a[0] != c[0] { 0 } else { 1 };
        if (a[axis] < b[axis] && b[axis] < c[axis]) || (c[axis] < b[axis] && b[axis] < a[axis]) {
            Ok(Walk::Through(point))
        } else {
            Err("a point lies inside an edge")
        }
    }
    /// The two triangles on either side of the edge between two points, and the vertex of each
    /// that is not on it.
    fn faces_on(&self, p: usize, q: usize) -> Option<(usize, usize, usize, usize)> {
        let mut found = (0..self.faces.len()).filter(|&f| {
            let nodes = self.faces[f].nodes;
            nodes.contains(&p) && nodes.contains(&q)
        });
        let (f, g) = (found.next()?, found.next()?);
        let off = |face: usize| (0..3).find(|&k| ![p, q].contains(&self.faces[face].nodes[k]));
        Some((f, off(f)?, g, off(g)?))
    }
}
