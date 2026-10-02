use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement,
        linear::{Hexahedron, Tetrahedron},
        mass::{IntegrationDensities, LumpedMassFiniteElement, MassFiniteElement},
    },
    math::{Quantity, Tensor},
    units::{Density, Mass, Volume},
};

const EPSILON: f64 = 1e-12;

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

mod consistent {
    use super::*;

    #[test]
    fn matches_the_closed_form_on_a_tetrahedron() {
        let element = Tetrahedron::<4>::from(tetrahedron_coordinates());
        let mass = DENSITY * element.volume();
        let masses = element.nodal_masses(&uniform());
        (0..4).for_each(|a| {
            (0..4).for_each(|b| {
                let expected = mass * (if a == b { 2.0 } else { 1.0 } / 20.0);
                assert!(!masses[a][b].differs(expected, EPSILON));
            })
        });
        let total = (0..4)
            .flat_map(|a| (0..4).map(move |b| (a, b)))
            .map(|(a, b)| masses[a][b])
            .sum::<Quantity<Mass>>();
        assert!(!total.differs(mass, EPSILON));
    }
}

mod lumped {
    use super::*;

    #[test]
    fn shares_a_tetrahedron_equally_with_one_point() {
        let element = Tetrahedron::<1>::from(tetrahedron_coordinates());
        let mass = DENSITY * element.volume();
        element
            .nodal_lumped_masses(&uniform())
            .iter()
            .for_each(|node_mass| assert!(!node_mass.differs(mass / 4.0, EPSILON)));
    }

    #[test]
    fn shares_a_tetrahedron_equally_with_four_points() {
        let element = Tetrahedron::<4>::from(tetrahedron_coordinates());
        let mass = DENSITY * element.volume();
        element
            .nodal_lumped_masses(&uniform())
            .iter()
            .for_each(|node_mass| assert!(!node_mass.differs(mass / 4.0, EPSILON)));
    }

    #[test]
    fn shares_a_hexahedron_equally() {
        let element = Hexahedron::from(hexahedron_coordinates());
        let mass = DENSITY * element.volume();
        assert!(!mass.differs(DENSITY * Volume::cubic_meters(6.0), EPSILON));
        element
            .nodal_lumped_masses(&uniform())
            .iter()
            .for_each(|node_mass| assert!(!node_mass.differs(mass / 8.0, EPSILON)));
    }

    #[test]
    fn is_the_row_sum_of_the_consistent_mass() {
        let element = Tetrahedron::<4>::from(tetrahedron_coordinates());
        let densities: IntegrationDensities<4> = [
            Density::kilograms_per_cubic_meter(1e3),
            Density::kilograms_per_cubic_meter(2e3),
            Density::kilograms_per_cubic_meter(3e3),
            Density::kilograms_per_cubic_meter(4e3),
        ]
        .into();
        let consistent = element.nodal_masses(&densities);
        element
            .nodal_lumped_masses(&densities)
            .iter()
            .zip(consistent.iter())
            .for_each(|(lumped, row)| {
                assert!(!lumped.differs(row.iter().copied().sum::<Quantity<Mass>>(), EPSILON))
            });
    }

    #[test]
    fn totals_the_mass_integrated_over_the_points() {
        let element = Tetrahedron::<4>::from(tetrahedron_coordinates());
        let densities: IntegrationDensities<4> = [
            Density::kilograms_per_cubic_meter(1e3),
            Density::kilograms_per_cubic_meter(2e3),
            Density::kilograms_per_cubic_meter(3e3),
            Density::kilograms_per_cubic_meter(4e3),
        ]
        .into();
        let expected = densities
            .iter()
            .zip(element.integration_weights())
            .map(|(density, integration_weight)| density * integration_weight)
            .sum::<Quantity<Mass>>();
        let total = element
            .nodal_lumped_masses(&densities)
            .iter()
            .copied()
            .sum::<Quantity<Mass>>();
        assert!(!total.differs(expected, EPSILON));
    }
}
