pub trait Sentinel: Copy {
    const MAX: Self;
}

impl Sentinel for u8 {
    const MAX: Self = u8::MAX;
}

impl Sentinel for u16 {
    const MAX: Self = u16::MAX;
}

impl Sentinel for u32 {
    const MAX: Self = u32::MAX;
}

impl Sentinel for u64 {
    const MAX: Self = u64::MAX;
}

impl Sentinel for usize {
    const MAX: Self = usize::MAX;
}
