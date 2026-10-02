use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, linear::TetrahedronConsistentMass,
        mass::MassFiniteElement,
    },
    math::Quantity,
    units::Volume,
};

#[test]
fn consistent_mass_matches_closed_form() {
    let element = TetrahedronConsistentMass::from(ElementNodalReferenceCoordinates::<4>::from([
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
    ]));
    let volume = element.volume();
    let masses = element.nodal_masses();
    (0..4).for_each(|a| {
        (0..4).for_each(|b| {
            let expected = volume * (if a == b { 2.0 } else { 1.0 } / 20.0);
            assert!(!masses[a][b].differs(expected, 1e-12));
        })
    });
    let total = (0..4)
        .flat_map(|a| (0..4).map(move |b| (a, b)))
        .map(|(a, b)| masses[a][b])
        .sum::<Quantity<Volume>>();
    assert!(!total.differs(volume, 1e-12));
}
