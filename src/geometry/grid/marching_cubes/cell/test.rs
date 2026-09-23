use super::Cell;

fn cell_with(corners: [f32; 8]) -> Cell {
    let mut cell = Cell::new(4, 4);
    cell.set_cube(0.0, [1, 2, 0], 1, corners);
    cell
}

#[test]
fn index_sets_a_bit_for_each_corner_above_the_level() {
    assert_eq!(cell_with([0.0; 8]).index, 0);
    assert_eq!(cell_with([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).index, 1);
    assert_eq!(
        cell_with([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).index,
        128
    );
    assert_eq!(cell_with([1.0; 8]).index, 255);
}

#[test]
fn index_subtracts_the_level() {
    let mut cell = Cell::new(2, 2);
    cell.set_cube(0.5, [0; 3], 1, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(cell.index, 2);
}

#[test]
fn shares_a_vertex_slot_between_neighbouring_cells() {
    let cell = cell_with([0.0; 8]);
    let origin = 4 * (4 * 2 + 1);
    assert_eq!(cell.face_layer_index(0), (0, origin));
    assert_eq!(cell.face_layer_index(1), (0, origin + 4 + 1));
    assert_eq!(cell.face_layer_index(2), (0, origin + 4 * 4));
    assert_eq!(cell.face_layer_index(3), (0, origin + 1));
    assert_eq!(cell.face_layer_index(4), (1, origin));
    assert_eq!(cell.face_layer_index(8), (0, origin + 2));
    assert_eq!(cell.face_layer_index(10), (0, origin + 4 * 5 + 2));
    assert_eq!(cell.face_layer_index(12), (0, origin + 3));
}

#[test]
fn a_new_layer_inherits_the_upper_layer_and_clears_the_next() {
    let mut cell = Cell::new(2, 2);
    cell.layers[0][3] = 9;
    cell.layers[1][3] = 7;
    cell.new_z_value();
    assert_eq!(cell.layers[0][3], 7);
    assert_eq!(cell.layers[1][3], -1);
}

#[test]
fn normals_are_unit_length_and_zero_stays_zero() {
    let mut cell = Cell::new(2, 2);
    let first = cell.add_vertex(0.0, 0.0, 0.0);
    cell.add_vertex(1.0, 1.0, 1.0);
    cell.add_gradient(first, [3.0, 0.0, 4.0]);
    let (_, _, normals, _) = cell.finish();
    assert_eq!(normals, vec![[0.6, 0.0, 0.8], [0.0; 3]]);
}

#[test]
fn a_vertex_keeps_the_largest_range_of_the_cells_that_use_it() {
    let mut cell = cell_with([1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    cell.prepare();
    let vertex = cell.add_vertex(0.0, 0.0, 0.0);
    cell.add_face(vertex);
    assert_eq!(cell.values[vertex], 4.0);
    cell.vmax = 2.0;
    cell.add_face(vertex);
    assert_eq!(cell.values[vertex], 4.0);
}

#[test]
fn the_centre_vertex_is_the_centre_of_a_symmetric_cube() {
    let mut cell = Cell::new(2, 2);
    cell.set_cube(
        0.0,
        [1, 2, 3],
        2,
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    );
    cell.prepare();
    let (position, _) = cell.center_vertex();
    assert_eq!(position, [2.0, 3.0, 4.0]);
}
