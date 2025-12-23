use crate::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block, Connectivity, FiniteElementBlock,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                GradientVectors,
                linear::wedge::{G, N, Wedge},
                solid::SolidFiniteElement,
                test::test_finite_element,
            },
            solid::{NodalForcesSolid, NodalStiffnessesSolid},
            test::test_finite_element_block,
        },
    },
    math::{ScalarList, Tensor, optimize::EqualityConstraint},
    mechanics::{DeformationGradient, DeformationGradientList, DeformationGradientRateList},
};

const D: usize = 21;

fn get_connectivity() -> Connectivity<N> {
    vec![
        [6, 0, 1, 13, 7, 8],
        [6, 5, 0, 13, 12, 7],
        [6, 4, 5, 13, 11, 12],
        [6, 3, 4, 13, 10, 11],
        [6, 2, 3, 13, 9, 10],
        [6, 1, 2, 13, 8, 9],
        [13, 7, 8, 20, 14, 15],
        [13, 12, 7, 20, 19, 14],
        [13, 11, 12, 20, 18, 19],
        [13, 10, 11, 20, 17, 18],
        [13, 9, 10, 20, 16, 17],
        [13, 8, 9, 20, 15, 16],
    ]
}

fn get_coordinates_block() -> NodalCoordinates {
    NodalCoordinates::from([
        [-0.09878799, -2.04153736, 0.04151086],
        [1.63754285, -0.99562875, -0.09887788],
        [1.78325268, 0.91336429, 0.0046213],
        [0.0315871, 1.98668723, -0.03387963],
        [-1.7616809, 1.05434027, -0.06391626],
        [-1.83054934, -0.918871, -0.06934134],
        [-0.04606216, 0.05635717, -0.06762617],
        [0.06188456, -1.94417605, 0.94833794],
        [1.77811662, -1.06366191, 0.99248902],
        [1.83070253, 0.90495846, 0.90698956],
        [0.04247115, 1.93735475, 0.94649422],
        [-1.68616882, 0.93665319, 0.98529049],
        [-1.79665063, -1.07254509, 1.01704746],
        [-0.06952031, -0.09988782, 1.02782312],
        [0.02586393, -2.00549432, 2.06038998],
        [1.66721894, -1.00169116, 1.92066057],
        [1.69052491, 0.96102527, 1.95475384],
        [-0.05315483, 2.0087481, 1.93177288],
        [-1.72731234, 0.90184272, 1.94313904],
        [-1.77668581, -1.01445848, 1.95960534],
        [-0.01508426, -0.09189985, 2.00903682],
    ])
}

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    ElementNodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
}

fn get_reference_coordinates_block() -> NodalReferenceCoordinates {
    NodalReferenceCoordinates::from([
        [0.0, -2.0, 0.0],
        [3.0_f64.sqrt(), -1.0, 0.0],
        [3.0_f64.sqrt(), 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [-3.0_f64.sqrt(), 1.0, 0.0],
        [-3.0_f64.sqrt(), -1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, -2.0, 1.0],
        [3.0_f64.sqrt(), -1.0, 1.0],
        [3.0_f64.sqrt(), 1.0, 1.0],
        [0.0, 2.0, 1.0],
        [-3.0_f64.sqrt(), 1.0, 1.0],
        [-3.0_f64.sqrt(), -1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, -2.0, 2.0],
        [3.0_f64.sqrt(), -1.0, 2.0],
        [3.0_f64.sqrt(), 1.0, 2.0],
        [0.0, 2.0, 2.0],
        [-3.0_f64.sqrt(), 1.0, 2.0],
        [-3.0_f64.sqrt(), -1.0, 2.0],
        [0.0, 0.0, 2.0],
    ])
}

fn get_velocities_block() -> NodalVelocities {
    NodalVelocities::from([
        [-0.09749712, -0.0025994, -0.06657566],
        [0.09572879, -0.05925453, 0.07497072],
        [-0.0683048, 0.0984304, 0.0454972],
        [-0.02008789, 0.07622423, -0.09598549],
        [0.06569026, -0.07759665, -0.05319748],
        [-0.0340313, -0.01151086, 0.08998172],
        [0.06716707, 0.03681826, 0.04703527],
        [-0.00529704, -0.00629051, 0.06642474],
        [0.05937476, -0.01342672, 0.04942817],
        [-0.04740402, 0.09363964, 0.04930639],
        [-0.05330261, -0.02076028, 0.05441078],
        [0.0685281, 0.01355718, 0.02223258],
        [0.06832807, 0.03535636, 0.03274881],
        [0.0686753, -0.02731762, 0.03348459],
    ])
}

fn equality_constraint() -> (
    crate::constitutive::solid::elastic::AppliedLoad,
    crate::math::Matrix,
    crate::math::Vector,
) {
    todo!()
}

fn applied_velocity(
    times: &crate::math::Vector,
) -> crate::constitutive::solid::viscoelastic::AppliedLoad<'_> {
    todo!()
}

fn applied_velocities() -> (crate::math::Matrix, crate::math::Vector) {
    todo!()
}

test_finite_element!(Wedge);
test_finite_element_block!(Wedge);
