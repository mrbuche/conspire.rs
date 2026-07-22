mod instance;
#[cfg(test)]
mod test;

pub(crate) use instance::Instance;

#[derive(Clone, Copy, PartialEq)]
enum Axis {
    Aligned,
    Overlap,
    Tangent,
    Disjoint,
}

fn axis(offset: i32) -> Axis {
    match offset.abs() {
        0 => Axis::Aligned,
        1 => Axis::Overlap,
        2 => Axis::Tangent,
        _ => Axis::Disjoint,
    }
}

pub(crate) fn conflicts<const D: usize>(offset: [i32; D]) -> bool {
    let axes = offset.map(axis);
    if axes.contains(&Axis::Disjoint) {
        return false;
    }
    if !axes.contains(&Axis::Tangent) {
        return true;
    }
    axes.contains(&Axis::Overlap)
}
