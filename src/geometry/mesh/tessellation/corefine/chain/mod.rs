#[cfg(test)]
mod test;

use super::{Crossing, Key};
use crate::{geometry::Coordinate, math::FxHashMap};

/// A curve along which two tessellations intersect, as a chain of crossing segments.
pub struct Polyline {
    /// The points along the curve, in order.
    pub points: Vec<Coordinate<3>>,
    /// Whether the curve closes on itself, joining the last point back to the first.
    pub closed: bool,
    /// The crossing each segment came from, segment `i` joining `points[i]` to the next point.
    pub crossings: Vec<usize>,
}

pub(super) fn chain(crossings: &[Crossing]) -> Result<Vec<Polyline>, &'static str> {
    let mut nodes = FxHashMap::<&Key, usize>::default();
    let mut points = Vec::<&Coordinate<3>>::new();
    let mut incident = Vec::<Vec<(usize, usize)>>::new();
    for (index, crossing) in crossings.iter().enumerate() {
        let ends = [0, 1].map(|end| {
            *nodes.entry(&crossing.keys[end]).or_insert_with(|| {
                points.push(&crossing.segment[end]);
                incident.push(Vec::new());
                points.len() - 1
            })
        });
        if ends[0] == ends[1] {
            return Err("a crossing segment starts and ends at the same point");
        }
        incident[ends[0]].push((index, ends[1]));
        incident[ends[1]].push((index, ends[0]));
    }
    if incident.iter().any(|segments| segments.len() > 2) {
        return Err("crossing curves branch");
    }
    let mut used = vec![false; crossings.len()];
    let mut polylines = Vec::new();
    for start in 0..points.len() {
        if incident[start].len() == 1 && !used[incident[start][0].0] {
            polylines.push(walk(start, &points, &incident, &mut used));
        }
    }
    for start in 0..points.len() {
        if incident[start].iter().any(|&(index, _)| !used[index]) {
            polylines.push(walk(start, &points, &incident, &mut used));
        }
    }
    Ok(polylines)
}

fn walk(
    start: usize,
    points: &[&Coordinate<3>],
    incident: &[Vec<(usize, usize)>],
    used: &mut [bool],
) -> Polyline {
    let mut polyline = Polyline {
        points: vec![points[start].clone()],
        closed: false,
        crossings: Vec::new(),
    };
    let mut node = start;
    while let Some(&(index, next)) = incident[node].iter().find(|&&(index, _)| !used[index]) {
        used[index] = true;
        polyline.crossings.push(index);
        if next == start {
            polyline.closed = true;
            break;
        }
        polyline.points.push(points[next].clone());
        node = next;
    }
    polyline
}
