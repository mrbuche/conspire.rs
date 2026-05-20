use crate::geometry::{
    Coordinate,
    bbox::{
        BoundingBox,
        unite::test::{BBOX_1, BBOX_1U2, BBOX_2, D, I},
    },
    bvh::item::Item,
};

type T = usize;

const ITEM_1: Item<D, I, T> = Item {
    bounding_box: BBOX_1,
    centroid: Coordinate::const_from([0.5, 0.5, 0.5]),
    index: 1,
};

const ITEM_2: Item<D, I, T> = Item {
    bounding_box: BBOX_2,
    centroid: Coordinate::const_from([1.5, 1.5, 1.5]),
    index: 2,
};

#[test]
fn bbox_from_items_slice() {
    let items = vec![ITEM_1, ITEM_2];
    let bbox = BoundingBox::from(items.as_slice());
    assert_eq!(bbox, BBOX_1U2)
}
