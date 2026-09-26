use super::Lut;

const LUT: Lut<12> = Lut {
    l1: 3,
    l2: 2,
    values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1],
};

#[test]
fn indexes_flat_tables() {
    assert_eq!(LUT.get1(7), 7);
}

#[test]
fn indexes_rows_of_a_table() {
    assert_eq!(LUT.get2(1, 2), 5);
}

#[test]
fn indexes_tables_of_tables() {
    assert_eq!(LUT.get3(1, 1, 1), 9);
}

#[test]
fn keeps_the_sign_of_padding() {
    assert_eq!(LUT.get1(11), -1);
}
