use crate::geometry::mesh::Connectivity;
use std::slice::Iter;

pub enum ElementIter<'a> {
    Hexahedral(Iter<'a, [usize; 8]>),
    Polyhedral(Iter<'a, Vec<usize>>),
    Polygonal(Iter<'a, Vec<usize>>),
    Pyramidal(Iter<'a, [usize; 5]>),
    Quadrilateral(Iter<'a, [usize; 4]>),
    Tetrahedral(Iter<'a, [usize; 4]>),
    Triangular(Iter<'a, [usize; 3]>),
    Wedge(Iter<'a, [usize; 6]>),
}

impl<'a> Iterator for ElementIter<'a> {
    type Item = &'a [usize];
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ElementIter::Hexahedral(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Polyhedral(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Polygonal(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Pyramidal(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Quadrilateral(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Tetrahedral(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Triangular(i) => i.next().map(|e| e.as_slice()),
            ElementIter::Wedge(i) => i.next().map(|e| e.as_slice()),
        }
    }
}

impl<'a> IntoIterator for &'a Connectivity {
    type Item = &'a [usize];
    type IntoIter = ElementIter<'a>;
    fn into_iter(self) -> ElementIter<'a> {
        match self {
            Connectivity::Hexahedral(c) => ElementIter::Hexahedral(c.iter()),
            Connectivity::Polyhedral(c) => ElementIter::Polyhedral(c.iter()),
            Connectivity::Polygonal(c) => ElementIter::Polygonal(c.iter()),
            Connectivity::Pyramidal(c) => ElementIter::Pyramidal(c.iter()),
            Connectivity::Quadrilateral(c) => ElementIter::Quadrilateral(c.iter()),
            Connectivity::Tetrahedral(c) => ElementIter::Tetrahedral(c.iter()),
            Connectivity::Triangular(c) => ElementIter::Triangular(c.iter()),
            Connectivity::Wedge(c) => ElementIter::Wedge(c.iter()),
        }
    }
}
