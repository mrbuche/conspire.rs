use crate::{
    domain::density::DensityField,
    math::{
        Quantity,
        assert::{Assert, AssertionError},
    },
    mechanics::ReferenceCoordinate,
    units::Density,
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

fn point() -> ReferenceCoordinate {
    [2.0, 3.0, 5.0].into()
}

fn field() -> impl Fn(&ReferenceCoordinate) -> Quantity<Density> {
    |coordinate| Density::kilograms_per_cubic_meter(1e3 + 1e2 * coordinate[0].value())
}

mod constant {
    use super::*;
    #[test]
    fn is_the_same_everywhere() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(DENSITY.density(&point()), &DENSITY)
    }
    #[test]
    fn resolves_to_itself_without_building_anything() -> Result<(), AssertionError> {
        let resolved = DENSITY.resolve(|_| -> () { unreachable!() });
        Assert::default().eq_within_tols(resolved, &DENSITY)
    }
}

mod varying {
    use super::*;
    #[test]
    fn is_evaluated_where_it_is_asked() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            field().density(&point()),
            &Density::kilograms_per_cubic_meter(1.2e3),
        )
    }
    #[test]
    fn resolves_to_whatever_its_discretization_builds() {
        let resolved = field().resolve(|field| vec![field.density(&point()); 3]);
        assert_eq!(resolved.len(), 3);
        resolved.iter().for_each(|density| {
            assert!(!density.differs(Density::kilograms_per_cubic_meter(1.2e3), 1e-12))
        });
    }
}
