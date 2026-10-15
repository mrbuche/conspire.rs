#[cfg(test)]
mod test;

use crate::geometry::mesh::{PrimitiveConnectivity, connectivity::primitive::LocalFacets};

/// Elements that can each list their faces, as slices of nodes, oriented outward.
pub trait ElementsFaces {
    fn number_of_elements(&self) -> usize;
    fn element_faces(&self, element: usize) -> impl Iterator<Item = impl AsRef<[usize]>>;
}

impl ElementsFaces for Vec<Vec<Vec<usize>>> {
    fn number_of_elements(&self) -> usize {
        self.len()
    }
    fn element_faces(&self, element: usize) -> impl Iterator<Item = impl AsRef<[usize]>> {
        self[element].iter()
    }
}

impl<const N: usize> ElementsFaces for PrimitiveConnectivity<3, N>
where
    Self: LocalFacets,
{
    fn number_of_elements(&self) -> usize {
        self.iter().len()
    }
    fn element_faces(&self, element: usize) -> impl Iterator<Item = impl AsRef<[usize]>> {
        let nodes = self.element(element);
        Self::LOCAL_FACETS
            .iter()
            .map(|local| Self::facet(local, nodes))
    }
}

impl<S: ElementsFaces> ElementsFaces for &S {
    fn number_of_elements(&self) -> usize {
        (**self).number_of_elements()
    }
    fn element_faces(&self, element: usize) -> impl Iterator<Item = impl AsRef<[usize]>> {
        (**self).element_faces(element)
    }
}
