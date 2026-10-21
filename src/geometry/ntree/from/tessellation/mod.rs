#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList,
        bbox::BoundingBox,
        mesh::Tessellation,
        ntree::{
            Octree,
            balance::Balancing,
            node::{Kind, Node, cell::Cell, slot::Slot},
            pair::Pairing,
            rescale::Rescaling,
            sizing::{Sizing, curvature::CurvatureSizing, separation::SeparationSizing},
        },
    },
    math::{Quantity, Scalar, Tensor},
    units::{Area, Dimensionless, Length},
};
use std::array::from_fn;

const D: usize = 3;
const M: usize = 6;

/// The point of a segment closest to `point`.
fn closest_on(segment: &[Coordinate<D>; 2], point: &Coordinate<D>) -> Coordinate<D> {
    let along = &segment[1] - &segment[0];
    let length = &along * &along;
    if length == Quantity::<Area>::new(0.0) {
        return segment[0].clone();
    }
    let fraction = Quantity::<Dimensionless>::new(
        ((point - &segment[0]) * &along / length)
            .value()
            .clamp(0.0, 1.0),
    );
    &segment[0] + &(along * fraction)
}

impl<T, U> Octree<T, U>
where
    T: Cell,
    U: Slot,
{
    /// Builds an octree from a tessellation, refining cells where either the
    /// local thickness or the local curvature demands a smaller size.
    ///
    /// `scale` controls cells-per-thickness; `curvature` controls
    /// curvature-driven refinement independent of thickness (e.g. a sphere
    /// has ~constant thickness everywhere but can still demand refinement
    /// from curvature alone). `separation` controls refinement driven by
    /// narrow features caught via nearby, topologically unrelated creases,
    /// independent of both thickness and curvature. `padding` adds extra
    /// empty root levels in case the tessellation's boundary overlaps the
    /// primordial primal node.
    pub fn from_features(
        tessellation: &Tessellation,
        scale: Scalar,
        curvature: CurvatureSizing,
        separation: SeparationSizing,
        padding: u16,
    ) -> Result<Self, &'static str> {
        Self::refine(&Sizing::new(
            tessellation,
            scale,
            curvature,
            separation,
            padding,
        ))
    }
    /// Refines an octree to a given size field.
    pub fn refine(sizing: &Sizing) -> Result<Self, &'static str> {
        let Sizing {
            center,
            coordinates,
            elements,
            levels,
            min_length,
            narrow,
            ratio,
            scale,
            targets,
        } = sizing;
        let (center, min_length, scale) = (center, *min_length, *scale);
        if elements.is_empty() {
            return Ok(Self {
                balanced: Balancing::None,
                nodes: vec![Node {
                    corner: from_fn(|_| T::ZERO),
                    length: T::ONE,
                    facets: [None; M],
                    kind: Kind::Leaf,
                    value: None,
                }],
                paired: Pairing::None,
                rescale: Rescaling {
                    center: Coordinate::const_from([0.0; D]),
                    cell: Quantity::new(1.0),
                    half: 0.0,
                },
            });
        }
        let root_length = 1usize
            .checked_shl(*levels)
            .and_then(T::length)
            .ok_or("sizing field exceeds maximum octree depth")?;
        let half = root_length.scalar() / 2.0;
        let mut tree = Self {
            balanced: Balancing::None,
            rescale: Rescaling {
                center: center.clone(),
                cell: min_length,
                half,
            },
            nodes: vec![Node {
                corner: from_fn(|_| T::ZERO),
                length: root_length,
                facets: [None; M],
                kind: Kind::Leaf,
                value: None,
            }],
            paired: Pairing::None,
        };
        let overlaps = |bbox: &BoundingBox<3>, triangle: usize| {
            let element = elements[triangle];
            bbox.overlaps_triangle(
                &coordinates[element[0]],
                &coordinates[element[1]],
                &coordinates[element[2]],
            )
        };
        // How far apart the two creases of a pair are where a cell sits, if
        // the cell lies between them: each crease within that distance, plus
        // `slack` times the cell's reach, of the cell's centre. The gap is
        // measured at the cell rather than carried by the pair, since a pair
        // running apart is narrow only at its closest point.
        let facing =
            |pair: usize, center: &Coordinate<3>, reach: Quantity<Length>, slack: Scalar| {
                let [one, two] = &narrow[pair];
                let (one, two) = (closest_on(one, center), closest_on(two, center));
                let gap = (&one - &two).norm();
                let near = gap + reach * slack;
                ((&one - center).norm() <= near && (&two - center).norm() <= near).then_some(gap)
            };
        let frame = |tree: &Self, node: usize| {
            let corner = tree.nodes[node].corner;
            let extent: Scalar = tree.nodes[node].length.scalar();
            let minimum = Coordinate::<3>::from(from_fn::<_, 3, _>(|ax| {
                center[ax] + min_length * (corner[ax].scalar() - half)
            }));
            let maximum =
                Coordinate::<3>::from(from_fn::<_, 3, _>(|ax| minimum[ax] + min_length * extent));
            let middle =
                Coordinate::<3>::from(from_fn::<_, 3, _>(|ax| (minimum[ax] + maximum[ax]) / 2.0));
            let reach = (&maximum - &minimum).norm() / 2.0;
            (minimum, maximum, middle, reach)
        };
        let mut stack: Vec<(usize, Vec<usize>, Vec<usize>)> = vec![(
            0,
            (0..elements.len()).collect(),
            (0..narrow.len()).collect(),
        )];
        while let Some((index, overlapping, nearby)) = stack.pop() {
            let cells: usize = tree.nodes[index].length.cells();
            let extent: Scalar = tree.nodes[index].length.scalar();
            let (_, _, middle, reach) = frame(&tree, index);
            let target = overlapping
                .iter()
                .map(|&triangle| targets[triangle])
                .chain(
                    nearby
                        .iter()
                        .filter_map(|&pair| facing(pair, &middle, reach, 1.0))
                        .map(|gap| gap * *ratio),
                )
                .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
            if min_length * (extent * scale) <= target {
                continue;
            }
            if cells <= 1 {
                continue;
            }
            tree.subdivide(index)?;
            let children: Vec<usize> = tree.nodes[index]
                .orthants()
                .unwrap()
                .iter()
                .map(|&child| child.slot())
                .collect();
            for child in children {
                let (minimum, maximum, center_of, reach) = frame(&tree, child);
                let bbox = BoundingBox::from(CoordinateList::const_from([minimum, maximum]));
                let inside: Vec<usize> = overlapping
                    .iter()
                    .copied()
                    .filter(|&triangle| overlaps(&bbox, triangle))
                    .collect();
                if inside.is_empty() {
                    continue;
                }
                // The gap varies over a cell, and the centre sees it from the
                // middle, so a pair is carried down while any smaller cell
                // inside could still lie between its creases.
                let within: Vec<usize> = nearby
                    .iter()
                    .copied()
                    .filter(|&pair| facing(pair, &center_of, reach, 3.0).is_some())
                    .collect();
                stack.push((child, inside, within));
            }
        }
        Ok(tree)
    }
}
