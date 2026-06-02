use crate::geometry::bbox::{
    Unite,
    test::{BBOX_1, BBOX_1U2, BBOX_2},
};

#[test]
fn unite() {
    assert_eq!(BBOX_1.unite(BBOX_2), BBOX_1U2);
}

#[test]
fn unite_ref() {
    assert_eq!(BBOX_1.unite(&BBOX_2), BBOX_1U2);
}

#[test]
fn ref_unite() {
    assert_eq!((&BBOX_1).unite(BBOX_2), BBOX_1U2);
}

#[test]
fn ref_unite_ref() {
    assert_eq!((&BBOX_1).unite(&BBOX_2), BBOX_1U2);
}
