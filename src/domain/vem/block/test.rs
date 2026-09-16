use crate::{
    constitutive::solid::elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
    domain::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::test::{
            test_finite_element_block_with_elastic_constitutive_model,
            test_finite_element_block_with_hyperelastic_constitutive_model,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::Tensor,
    mechanics::{DeformationGradient, test::get_deformation_gradient},
    vem::block::{Block, element::Element},
};

const D: usize = 20;

fn get_element_face_connectivity() -> Vec<Vec<usize>> {
    vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
}

fn get_face_node_connectivity() -> Vec<Vec<usize>> {
    vec![
        vec![16, 17, 4, 8, 0],
        vec![12, 13, 2, 16, 0],
        vec![8, 9, 1, 12, 0],
        vec![9, 5, 19, 18, 1],
        vec![18, 3, 13, 12, 1],
        vec![10, 6, 17, 16, 2],
        vec![13, 3, 11, 10, 2],
        vec![7, 11, 3, 18, 19],
        vec![14, 5, 9, 8, 4],
        vec![6, 15, 14, 4, 17],
        vec![5, 14, 15, 7, 19],
        vec![6, 10, 11, 7, 15],
    ]
}

fn get_reference_coordinates_block() -> NodalReferenceCoordinates<3> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    NodalReferenceCoordinates::from([
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [0.0, -phi, -1.0 / phi],
        [0.0, -phi, 1.0 / phi],
        [0.0, phi, -1.0 / phi],
        [0.0, phi, 1.0 / phi],
        [-phi, -1.0 / phi, 0.0],
        [-phi, 1.0 / phi, 0.0],
        [phi, -1.0 / phi, 0.0],
        [phi, 1.0 / phi, 0.0],
        [-1.0 / phi, 0.0, -phi],
        [1.0 / phi, 0.0, -phi],
        [-1.0 / phi, 0.0, phi],
        [1.0 / phi, 0.0, phi],
    ])
}

fn get_coordinates_block() -> NodalCoordinates<3> {
    get_reference_coordinates_block()
        .iter()
        .map(|reference_coordinate| get_deformation_gradient() * reference_coordinate)
        .collect()
}

macro_rules! setup_block {
    ($constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_block() -> Block<$constitutive_model_type, Element> {
            Block::<$constitutive_model_type, Element>::from((
                $constitutive_model,
                get_element_face_connectivity(),
                get_face_node_connectivity(),
                &get_reference_coordinates_block(),
            ))
        }
        fn get_block_transformed() -> Block<$constitutive_model_type, Element> {
            Block::<$constitutive_model_type, Element>::from((
                $constitutive_model,
                get_element_face_connectivity(),
                get_face_node_connectivity(),
                &get_reference_coordinates_transformed_block(),
            ))
        }
    };
}

crate::domain::block::test::test_block_elastic_and_hyperelastic!(Element);
