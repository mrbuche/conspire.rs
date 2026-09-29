#[cfg(test)]
mod test;

use super::{BoundaryConditions, CornerSelection, Row, rigid::ModeCover};
use crate::geometry::mesh::Partition;
use std::collections::{HashMap, HashSet};

/// The corners of a partition, with those of its constraints.
///
/// The nodes shared by three or more subdomains are corners, and so are all
/// those of a constraint of several entries. A constraint of one entry, which
/// prescribes a displacement, is left to the interface operator, so that the
/// corner problem does not grow with the number of them, except that a few of
/// their nodes are made corners where needed. A subdomain needs enough of its
/// DOFs held to remove its rigid-body modes, and the whole block enough
/// constrained DOFs to remove its own, and neither is met by the multipliers
/// of the interface operator.
pub(crate) fn select_corners<const D: usize>(
    partition: &Partition,
    positions: &[[f64; D]],
    boundary_conditions: &BoundaryConditions,
    rows: &[Row],
) -> CornerSelection {
    let removable = D + D * (D - 1) / 2;
    let mut corners: HashSet<usize> = CornerSelection::from_partition(partition)
        .nodes()
        .iter()
        .copied()
        .collect();
    rows.iter()
        .filter(|row| row.single().is_none())
        .for_each(|row| {
            row.entries.iter().for_each(|&(node, _, _)| {
                corners.insert(node);
            })
        });
    let mut prescribed = HashMap::<usize, Vec<usize>>::new();
    rows.iter()
        .filter_map(Row::single)
        .for_each(|(node, component, _)| prescribed.entry(node).or_default().push(component));
    let mut fixed = HashMap::<usize, Vec<usize>>::new();
    boundary_conditions
        .fixed()
        .for_each(|(node, component)| fixed.entry(node).or_default().push(component));
    partition.parts_nodes().iter().for_each(|nodes| {
        if nodes.is_empty() {
            return;
        }
        let local: Vec<[f64; D]> = nodes.iter().map(|&node| positions[node]).collect();
        let mut cover = ModeCover::new(&local);
        nodes.iter().enumerate().for_each(|(index, &node)| {
            if corners.contains(&node) {
                (0..D).for_each(|component| {
                    cover.add(D * index + component);
                })
            } else if let Some(components) = fixed.get(&node) {
                components.iter().for_each(|&component| {
                    cover.add(D * index + component);
                })
            }
        });
        while cover.rank() < removable {
            let best = nodes
                .iter()
                .enumerate()
                .filter(|&(_, node)| !corners.contains(node) && prescribed.contains_key(node))
                .map(|(index, &node)| {
                    let gain = (0..D)
                        .map(|component| cover.gain(D * index + component))
                        .fold(0.0, f64::max);
                    (gain, index, node)
                })
                .max_by(|a, b| a.0.total_cmp(&b.0));
            match best {
                Some((gain, index, node)) if gain > 0.0 => {
                    (0..D).for_each(|component| {
                        cover.add(D * index + component);
                    });
                    corners.insert(node);
                }
                _ => break,
            }
        }
    });
    let mut cover = ModeCover::new(positions);
    boundary_conditions.fixed().for_each(|(node, component)| {
        cover.add(D * node + component);
    });
    rows.iter()
        .filter_map(Row::single)
        .filter(|(node, _, _)| corners.contains(node))
        .for_each(|(node, component, _)| {
            cover.add(D * node + component);
        });
    while cover.rank() < removable {
        let best = rows
            .iter()
            .filter_map(Row::single)
            .filter(|(node, _, _)| !corners.contains(node))
            .map(|(node, component, _)| (cover.gain(D * node + component), node))
            .max_by(|a, b| a.0.total_cmp(&b.0));
        match best {
            Some((gain, node)) if gain > 0.0 => {
                corners.insert(node);
                prescribed[&node].iter().for_each(|&component| {
                    cover.add(D * node + component);
                })
            }
            _ => break,
        }
    }
    CornerSelection::new(corners.into_iter().collect())
}
