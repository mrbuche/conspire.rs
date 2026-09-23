#[cfg(test)]
mod test;

use crate::{geometry::mesh::Mesh, math::Tensor};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bisection {
    Coordinate,
    Principal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PartitionQuality {
    pub sizes: Vec<usize>,
    pub imbalance: f64,
    pub interface_nodes: usize,
    pub disconnected_parts: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Partition {
    elements_parts: Vec<usize>,
    parts_elements: Vec<Vec<usize>>,
    parts_nodes: Vec<Vec<usize>>,
    nodes_parts: Vec<Vec<usize>>,
}

impl Partition {
    pub fn new<const D: usize>(mesh: &Mesh<D>, elements_parts: Vec<usize>) -> Self {
        assert_eq!(
            elements_parts.len(),
            mesh.number_of_elements(),
            "assignment must have one entry per element"
        );
        let number_of_parts = elements_parts
            .iter()
            .map(|&part| part + 1)
            .max()
            .unwrap_or(0);
        let mut parts_elements = vec![Vec::new(); number_of_parts];
        elements_parts
            .iter()
            .enumerate()
            .for_each(|(element, &part)| parts_elements[part].push(element));
        let mut parts_nodes = vec![Vec::new(); number_of_parts];
        mesh.element_nodes()
            .zip(&elements_parts)
            .for_each(|(nodes, &part)| parts_nodes[part].extend(nodes));
        parts_nodes.iter_mut().for_each(|nodes| {
            nodes.sort_unstable();
            nodes.dedup();
        });
        let mut nodes_parts = vec![Vec::new(); mesh.number_of_nodes()];
        parts_nodes
            .iter()
            .enumerate()
            .for_each(|(part, nodes)| nodes.iter().for_each(|&node| nodes_parts[node].push(part)));
        Self {
            elements_parts,
            parts_elements,
            parts_nodes,
            nodes_parts,
        }
    }
    pub fn number_of_parts(&self) -> usize {
        self.parts_elements.len()
    }
    pub fn elements_parts(&self) -> &[usize] {
        &self.elements_parts
    }
    pub fn part_of(&self, element: usize) -> usize {
        self.elements_parts[element]
    }
    pub fn part_elements(&self, part: usize) -> &[usize] {
        &self.parts_elements[part]
    }
    pub fn part_nodes(&self, part: usize) -> &[usize] {
        &self.parts_nodes[part]
    }
    pub fn parts_nodes(&self) -> &[Vec<usize>] {
        &self.parts_nodes
    }
    pub fn node_parts(&self, node: usize) -> &[usize] {
        &self.nodes_parts[node]
    }
    pub fn interface_nodes(&self) -> Vec<usize> {
        self.nodes_parts
            .iter()
            .enumerate()
            .filter(|(_, parts)| parts.len() > 1)
            .map(|(node, _)| node)
            .collect()
    }
    pub fn part_mesh<const D: usize>(&self, mesh: &Mesh<D>, part: usize) -> (Mesh<D>, Vec<usize>) {
        mesh.retained_elements(|element, _, _| self.elements_parts[element] == part)
    }
    pub fn quality<const D: usize>(&self, mesh: &Mesh<D>) -> PartitionQuality {
        let sizes = self.parts_elements.iter().map(Vec::len).collect::<Vec<_>>();
        let mean = self.elements_parts.len() as f64 / sizes.len() as f64;
        let mut roots = (0..self.elements_parts.len()).collect::<Vec<_>>();
        mesh.node_element_connectivity()
            .iter()
            .for_each(|elements| {
                let mut firsts: Vec<(usize, usize)> = Vec::new();
                elements.iter().for_each(|&element| {
                    match firsts
                        .iter()
                        .find(|&&(part, _)| part == self.elements_parts[element])
                    {
                        Some(&(_, first)) => {
                            let (a, b) = (find(&mut roots, first), find(&mut roots, element));
                            roots[b] = a;
                        }
                        None => firsts.push((self.elements_parts[element], element)),
                    }
                })
            });
        let mut components = vec![0usize; sizes.len()];
        (0..self.elements_parts.len())
            .filter(|&element| find(&mut roots, element) == element)
            .for_each(|element| components[self.elements_parts[element]] += 1);
        PartitionQuality {
            imbalance: sizes.iter().copied().max().unwrap_or(0) as f64 / mean,
            interface_nodes: self.interface_nodes().len(),
            disconnected_parts: components.iter().filter(|&&count| count > 1).count(),
            sizes,
        }
    }
}

impl<const D: usize> Mesh<D> {
    pub fn partition_rcb(&self, parts: usize) -> Partition {
        self.partition_bisection(parts, Bisection::Coordinate)
    }
    pub fn partition_rib(&self, parts: usize) -> Partition {
        self.partition_bisection(parts, Bisection::Principal)
    }
    pub fn partition_bisection(&self, parts: usize, bisection: Bisection) -> Partition {
        let points = self.element_points();
        assert!(
            (1..=points.len()).contains(&parts),
            "parts must be between 1 and the number of elements"
        );
        let mut assignment = vec![0; points.len()];
        let mut elements = (0..points.len()).collect::<Vec<_>>();
        bisect(&mut elements, parts, 0, &points, bisection, &mut assignment);
        Partition::new(self, assignment)
    }
    pub fn partition_box(&self, divisions: [usize; D]) -> Partition {
        assert!(
            divisions.iter().all(|&n| n > 0),
            "divisions must be positive"
        );
        let points = self.element_points();
        let mut lower = [f64::INFINITY; D];
        let mut upper = [f64::NEG_INFINITY; D];
        points.iter().for_each(|point| {
            (0..D).for_each(|axis| {
                lower[axis] = lower[axis].min(point[axis]);
                upper[axis] = upper[axis].max(point[axis]);
            })
        });
        let assignment = points
            .iter()
            .map(|point| {
                (0..D).rev().fold(0, |part, axis| {
                    let extent = upper[axis] - lower[axis];
                    let cell = if extent > 0.0 {
                        (((point[axis] - lower[axis]) / extent * divisions[axis] as f64) as usize)
                            .min(divisions[axis] - 1)
                    } else {
                        0
                    };
                    part * divisions[axis] + cell
                })
            })
            .collect();
        Partition::new(self, assignment)
    }
    fn element_points(&self) -> Vec<[f64; D]> {
        self.centroids()
            .iter()
            .map(|centroid| std::array::from_fn(|axis| centroid[axis].value()))
            .collect()
    }
    fn element_nodes(&self) -> impl Iterator<Item = Vec<usize>> {
        self.iter().flat_map(|block| {
            block
                .iter()
                .map(move |element| block.element_nodes(element))
        })
    }
}

fn find(roots: &mut [usize], mut element: usize) -> usize {
    while roots[element] != element {
        roots[element] = roots[roots[element]];
        element = roots[element];
    }
    element
}

fn bisect<const D: usize>(
    elements: &mut [usize],
    parts: usize,
    first: usize,
    points: &[[f64; D]],
    bisection: Bisection,
    assignment: &mut [usize],
) {
    if parts == 1 {
        elements
            .iter()
            .for_each(|&element| assignment[element] = first);
        return;
    }
    let lower = parts / 2;
    let split = elements.len() * lower / parts;
    let direction = match bisection {
        Bisection::Coordinate => widest_axis(elements, points),
        Bisection::Principal => principal_axis(elements, points),
    };
    let key = |element: usize| -> f64 {
        (0..D)
            .map(|axis| direction[axis] * points[element][axis])
            .sum()
    };
    elements.select_nth_unstable_by(split, |&a, &b| {
        key(a)
            .partial_cmp(&key(b))
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    let (left, right) = elements.split_at_mut(split);
    bisect(left, lower, first, points, bisection, assignment);
    bisect(
        right,
        parts - lower,
        first + lower,
        points,
        bisection,
        assignment,
    );
}

fn widest_axis<const D: usize>(elements: &[usize], points: &[[f64; D]]) -> [f64; D] {
    let extent = |axis: usize| {
        let (low, high) = elements.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(low, high), &element| {
                (
                    low.min(points[element][axis]),
                    high.max(points[element][axis]),
                )
            },
        );
        high - low
    };
    let widest = (1..D).fold(0, |best, axis| {
        if extent(axis) > extent(best) {
            axis
        } else {
            best
        }
    });
    std::array::from_fn(|index| if index == widest { 1.0 } else { 0.0 })
}

fn principal_axis<const D: usize>(elements: &[usize], points: &[[f64; D]]) -> [f64; D] {
    let count = elements.len() as f64;
    let mean: [f64; D] = std::array::from_fn(|axis| {
        elements
            .iter()
            .map(|&element| points[element][axis])
            .sum::<f64>()
            / count
    });
    let mut covariance = [[0.0; D]; D];
    elements.iter().for_each(|&element| {
        (0..D).for_each(|row| {
            (0..D).for_each(|column| {
                covariance[row][column] +=
                    (points[element][row] - mean[row]) * (points[element][column] - mean[column])
            })
        })
    });
    let mut direction: [f64; D] = std::array::from_fn(|axis| 1.0 + 0.1 * axis as f64);
    (0..64).for_each(|_| {
        let next: [f64; D] = std::array::from_fn(|row| {
            (0..D)
                .map(|column| covariance[row][column] * direction[column])
                .sum()
        });
        let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            direction = std::array::from_fn(|axis| next[axis] / norm);
        }
    });
    if direction.iter().all(|x| x.is_finite()) && direction.iter().any(|&x| x != 0.0) {
        direction
    } else {
        widest_axis(elements, points)
    }
}
