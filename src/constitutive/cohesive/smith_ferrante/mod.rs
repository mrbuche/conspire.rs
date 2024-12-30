#[cfg(test)]
pub mod test;

use super::*;

/// The Smith-Ferrante cohesive constitutive model.
pub struct SmithFerrante<'a> {
    parameters: Parameters<'a>,
}

impl SmithFerrante<'_> {
    fn get_characteristic_displacement(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn get_maximum_normal_traction(&self) -> &Scalar {
        &self.parameters[1]
    }
    fn get_weight(&self) -> &Scalar {
        &self.parameters[2]
    }
}

impl<'a> Constitutive<'a> for SmithFerrante<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Cohesive<'a> for SmithFerrante<'a> {
    fn calculate_traction(&self, displacement: &Displacement, normal: &Normal) -> Traction {
        let opening_magnitude = displacement * normal;
        let opening = normal * opening_magnitude;
        let sliding = displacement - normal * opening_magnitude;
        let effective_opening = sliding * self.get_weight() + opening;
        let effective_opening_magnitude = effective_opening.norm();
        effective_opening
            * (self.get_maximum_normal_traction() / self.get_characteristic_displacement())
            * (1.0 - effective_opening_magnitude / self.get_characteristic_displacement()).exp()
    }
    fn calculate_stiffnesses(
        &self,
        displacement: &Displacement,
        normal: &Normal,
    ) -> (Stiffness, Stiffness) {
        let opening_magnitude = displacement * normal;
        let opening = normal * opening_magnitude;
        let sliding = displacement - normal * opening_magnitude;
        let effective_opening = sliding * self.get_weight() + opening;
        let nominal = (IDENTITY
            - Stiffness::dyad(&effective_opening, &effective_opening.copy().normalized())
                / self.get_characteristic_displacement())
            * ((self.get_maximum_normal_traction() / self.get_characteristic_displacement())
                * (1.0 - effective_opening.norm() / self.get_characteristic_displacement()).exp());
        (
            &nominal
                * (IDENTITY * self.get_weight()
                    + Stiffness::dyad(normal, normal) * (1.0 - self.get_weight())),
            (&nominal
                * ((IDENTITY * opening_magnitude + Stiffness::dyad(normal, displacement))
                    * (1.0 - self.get_weight()))),
        )
    }
}
