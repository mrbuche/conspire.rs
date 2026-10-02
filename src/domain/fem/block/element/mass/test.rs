use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement,
        linear::{Hexahedron, Tetrahedron},
        mass::{
            ElementNodalLumpedMasses, ElementNodalMasses, IntegrationDensities,
            LumpedMassFiniteElement, MassFiniteElement,
        },
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::{Density, Mass, Volume},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

fn tetrahedron_coordinates() -> ElementNodalReferenceCoordinates<4> {
    ElementNodalReferenceCoordinates::from([
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
    ])
}

fn hexahedron_coordinates() -> ElementNodalReferenceCoordinates<8> {
    ElementNodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
        [2.0, 0.0, 3.0],
        [2.0, 1.0, 3.0],
        [0.0, 1.0, 3.0],
    ])
}

fn uniform<const G: usize>() -> IntegrationDensities<G> {
    [DENSITY; G].into()
}

fn varying() -> IntegrationDensities<4> {
    [
        Density::kilograms_per_cubic_meter(1e3),
        Density::kilograms_per_cubic_meter(2e3),
        Density::kilograms_per_cubic_meter(3e3),
        Density::kilograms_per_cubic_meter(4e3),
    ]
    .into()
}

mod consistent {
    use super::*;

    #[test]
    fn matches_the_closed_form_on_a_tetrahedron() -> Result<(), AssertionError> {
        let element = Tetrahedron::<4>::from(tetrahedron_coordinates());
        let mass = DENSITY * element.volume();
        let expected: ElementNodalMasses<4> = (0..4)
            .map(|a| {
                (0..4)
                    .map(|b| mass * (if a == b { 2.0 } else { 1.0 } / 20.0))
                    .collect()
            })
            .collect();
        let masses = element.nodal_masses(&uniform());
        Assert::default().eq_within_tols(&masses, &expected)?;
        let total = masses
            .iter()
            .flat_map(|row| row.iter().copied())
            .sum::<Quantity<Mass>>();
        Assert::default().eq_within_tols(total, &mass)
    }
}

mod lumped {
    use super::*;

    #[test]
    fn shares_a_tetrahedron_equally_with_one_point() -> Result<(), AssertionError> {
        let element = Tetrahedron::<1>::from(tetrahedron_coordinates());
        let expected: ElementNodalLumpedMasses<4> = [DENSITY * element.volume() / 4.0; 4].into();
        Assert::default().eq_within_tols(element.nodal_lumped_masses(&uniform()), &expected)
    }

    #[test]
    fn shares_a_tetrahedron_equally_with_four_points() -> Result<(), AssertionError> {
        let element = Tetrahedron::<4>::from(tetrahedron_coordinates());
        let expected: ElementNodalLumpedMasses<4> = [DENSITY * element.volume() / 4.0; 4].into();
        Assert::default().eq_within_tols(element.nodal_lumped_masses(&uniform()), &expected)
    }

    #[test]
    fn shares_a_hexahedron_equally() -> Result<(), AssertionError> {
        let element = Hexahedron::from(hexahedron_coordinates());
        let mass = DENSITY * element.volume();
        Assert::default().eq_within_tols(mass, &(DENSITY * Volume::cubic_meters(6.0)))?;
        let expected: ElementNodalLumpedMasses<8> = [mass / 8.0; 8].into();
        Assert::default().eq_within_tols(element.nodal_lumped_masses(&uniform()), &expected)
    }

    #[test]
    fn is_the_row_sum_of_the_consistent_mass() -> Result<(), AssertionError> {
        let element = Tetrahedron::<4>::from(tetrahedron_coordinates());
        let row_sums: ElementNodalLumpedMasses<4> = element
            .nodal_masses(&varying())
            .iter()
            .map(|row| row.iter().copied().sum::<Quantity<Mass>>())
            .collect();
        Assert::default().eq_within_tols(element.nodal_lumped_masses(&varying()), &row_sums)
    }

    #[test]
    fn totals_the_mass_integrated_over_the_points() -> Result<(), AssertionError> {
        let element = Tetrahedron::<4>::from(tetrahedron_coordinates());
        let expected = varying()
            .iter()
            .zip(element.integration_weights())
            .map(|(density, integration_weight)| density * integration_weight)
            .sum::<Quantity<Mass>>();
        let total = element
            .nodal_lumped_masses(&varying())
            .iter()
            .copied()
            .sum::<Quantity<Mass>>();
        Assert::default().eq_within_tols(total, &expected)
    }
}
