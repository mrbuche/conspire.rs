use crate::{
    geometry::mesh::test::{square, tetrahedra},
    math::Quantity,
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

#[test]
fn rows_sum_to_the_integral_of_each_function() {
    let mesh = tetrahedra(6);
    let seeds = mesh.sample(length(0.4), 2);
    let basis = mesh.reproducing_basis(&seeds, length(1.1), 1, 1).unwrap();
    let products = mesh.inner_products(&basis).unwrap();
    let integrals = mesh.integrals(&basis).unwrap();
    products.iter().zip(&integrals).for_each(|(row, integral)| {
        let sum: f64 = row.iter().map(|(_, product)| product.value()).sum();
        assert!((sum - integral.value()).abs() < 1e-12);
    });
    let total: f64 = products
        .iter()
        .flatten()
        .map(|(_, product)| product.value())
        .sum();
    assert!((total - 1.0).abs() < 1e-12, "{total}");
}

#[test]
fn products_are_symmetric_and_sorted() {
    let mesh = tetrahedra(6);
    let seeds = mesh.sample(length(0.4), 2);
    let basis = mesh.reproducing_basis(&seeds, length(1.1), 1, 1).unwrap();
    let products = mesh.inner_products(&basis).unwrap();
    products.iter().enumerate().for_each(|(i, row)| {
        assert!(row.windows(2).all(|pair| pair[0].0 < pair[1].0));
        row.iter().for_each(|&(j, product)| {
            let (_, transpose) = products[j]
                .iter()
                .find(|&&(k, _)| k == i)
                .expect("symmetric pattern");
            assert!((product.value() - transpose.value()).abs() < 1e-12);
        });
    });
}

#[test]
fn products_in_two_dimensions() {
    let mesh = square(40);
    let seeds = mesh.sample(length(0.2), 3);
    let basis = mesh.reproducing_basis(&seeds, length(0.52), 1, 1).unwrap();
    let products = mesh.inner_products(&basis).unwrap();
    let total: f64 = products
        .iter()
        .flatten()
        .map(|(_, product)| product.value())
        .sum();
    assert!((total - 1.0).abs() < 1e-12, "{total}");
}

#[test]
fn a_function_has_a_positive_self_product() {
    let mesh = tetrahedra(6);
    let seeds = mesh.sample(length(0.4), 2);
    let basis = mesh.reproducing_basis(&seeds, length(1.1), 1, 1).unwrap();
    let products = mesh.inner_products(&basis).unwrap();
    products.iter().enumerate().for_each(|(i, row)| {
        let (_, own) = row.iter().find(|&&(j, _)| j == i).unwrap();
        assert!(own.value() > 0.0);
    });
}
