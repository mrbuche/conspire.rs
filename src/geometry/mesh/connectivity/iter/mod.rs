use crate::geometry::mesh::Connectivity;
use std::slice::Iter;

pub enum ElementIter<'a> {
    Hexahedral(Iter<'a, [usize; 8]>),
    Polyhedral(Iter<'a, Vec<usize>>),
    Polygonal(Iter<'a, Vec<usize>>),
    Quadrilateral(Iter<'a, [usize; 4]>),
    Tetrahedral(Iter<'a, [usize; 4]>),
    Triangular(Iter<'a, [usize; 3]>),
}

impl<'a> Iterator for ElementIter<'a> {
    type Item = &'a [usize];
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ElementIter::Hexahedral(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Polyhedral(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Polygonal(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Quadrilateral(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Tetrahedral(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Triangular(i) => i.next().map(|e| e.as_slice()),
        }
    }
}

impl<'a> IntoIterator for &'a Connectivity {
    type Item = &'a [usize];
    type IntoIter = ElementIter<'a>;
    fn into_iter(self) -> ElementIter<'a> {
        match self {
            Connectivity::Hexahedral(c) => ElementIter::Hexahedral(c.0.iter()),
            Connectivity::Polyhedral(c) => ElementIter::Polyhedral(c.0.iter()),
            Connectivity::Polygonal(c) => ElementIter::Polygonal(c.0.iter()),
            Connectivity::Quadrilateral(c) => ElementIter::Quadrilateral(c.0.iter()),
            Connectivity::Tetrahedral(c) => ElementIter::Tetrahedral(c.0.iter()),
            Connectivity::Triangular(c) => ElementIter::Triangular(c.0.iter()),
        }
    }
}
