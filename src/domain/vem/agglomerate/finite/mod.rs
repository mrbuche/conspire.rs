use super::Candidates;
use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates,
        linear::{Hexahedron, Pyramid, Tetrahedron, Wedge},
        mass::{ElementNodalLumpedMasses, IntegrationDensities, LumpedMassFiniteElement},
        solid::{
            ElementNodalStiffnessesSolid,
            elastic::{ElasticElement, ElasticFiniteElement},
        },
        time_scale::{fastest_time_scale, time_scale_exceeds},
    },
    geometry::mesh::{Connectivity, Mesh},
    math::Quantity,
    units::{Density, Time},
};

/// An element that, if left alone, is run as a finite element.
#[derive(Clone, Copy)]
pub(super) enum Primitive {
    Hexahedron([usize; 8]),
    Pyramid([usize; 5]),
    Tetrahedron([usize; 4]),
    Wedge([usize; 6]),
    Virtual,
}

impl Primitive {
    pub(super) fn from_mesh(mesh: &Mesh<3>) -> Vec<Self> {
        mesh.iter()
            .flat_map(|block| match block {
                Connectivity::Hexahedral(c) => c.iter().map(|&n| Self::Hexahedron(n)).collect(),
                Connectivity::Pyramidal(c) => c.iter().map(|&n| Self::Pyramid(n)).collect(),
                Connectivity::Tetrahedral(c) => c.iter().map(|&n| Self::Tetrahedron(n)).collect(),
                Connectivity::Wedge(c) => c.iter().map(|&n| Self::Wedge(n)).collect(),
                _ => vec![Self::Virtual; block.number_of_elements()],
            })
            .collect()
    }
}

impl<S> Candidates<S> {
    fn primitive(&self, elements: &[usize]) -> Option<Primitive> {
        match elements {
            [element] => self
                .primitives
                .get(*element)
                .copied()
                .filter(|primitive| !matches!(primitive, Primitive::Virtual)),
            _ => None,
        }
    }
    /// The time scale of an element on its own as the finite element it is.
    pub(super) fn finite_time_scale(
        &self,
        elements: &[usize],
    ) -> Option<Result<Quantity<Time>, String>> {
        Some(match self.primitive(elements)? {
            Primitive::Hexahedron(n) => self
                .finite_matrices::<Hexahedron, 8, 8>(&n)
                .map(|(k, m)| fastest_time_scale(&k, &m)),
            Primitive::Pyramid(n) => self
                .finite_matrices::<Pyramid, 8, 5>(&n)
                .map(|(k, m)| fastest_time_scale(&k, &m)),
            Primitive::Tetrahedron(n) => self
                .finite_matrices::<Tetrahedron<1>, 1, 4>(&n)
                .map(|(k, m)| fastest_time_scale(&k, &m)),
            Primitive::Wedge(n) => self
                .finite_matrices::<Wedge, 6, 6>(&n)
                .map(|(k, m)| fastest_time_scale(&k, &m)),
            Primitive::Virtual => return None,
        })
    }
    /// Whether the time scale of an element on its own as the finite element it is is certified.
    pub(super) fn finite_time_scale_exceeds(
        &self,
        elements: &[usize],
        minimum: Quantity<Time>,
    ) -> Option<Result<bool, String>> {
        Some(match self.primitive(elements)? {
            Primitive::Hexahedron(n) => self
                .finite_matrices::<Hexahedron, 8, 8>(&n)
                .map(|(k, m)| time_scale_exceeds(&k, &m, minimum)),
            Primitive::Pyramid(n) => self
                .finite_matrices::<Pyramid, 8, 5>(&n)
                .map(|(k, m)| time_scale_exceeds(&k, &m, minimum)),
            Primitive::Tetrahedron(n) => self
                .finite_matrices::<Tetrahedron<1>, 1, 4>(&n)
                .map(|(k, m)| time_scale_exceeds(&k, &m, minimum)),
            Primitive::Wedge(n) => self
                .finite_matrices::<Wedge, 6, 6>(&n)
                .map(|(k, m)| time_scale_exceeds(&k, &m, minimum)),
            Primitive::Virtual => return None,
        })
    }
    fn finite_matrices<F, const G: usize, const N: usize>(
        &self,
        nodes: &[usize; N],
    ) -> Result<(ElementNodalStiffnessesSolid<N>, ElementNodalLumpedMasses<N>), String>
    where
        F: ElasticFiniteElement<NeoHookean, G, 3, N, N>
            + LumpedMassFiniteElement<G, 3, N, N>
            + From<ElementNodalReferenceCoordinates<N>>,
    {
        let reference: ElementNodalReferenceCoordinates<N> = nodes
            .iter()
            .map(|&node| self.coordinates[node].clone())
            .collect();
        let current: ElementNodalCoordinates<N> = nodes
            .iter()
            .map(|&node| self.coordinates[node].clone().into())
            .collect();
        let element = F::from(reference);
        let stiffnesses = ElasticElement::nodal_stiffnesses(&element, &self.material, &current)
            .map_err(|error| format!("{error:?}"))?;
        let densities: IntegrationDensities<G> =
            [Density::kilograms_per_cubic_meter(1.0); G].into();
        Ok((
            stiffnesses,
            LumpedMassFiniteElement::nodal_lumped_masses(&element, &densities),
        ))
    }
}
