#[cfg(test)]
mod test;

fn rot(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= rot(*c, 4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= rot(*a, 6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= rot(*b, 8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= rot(*c, 16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= rot(*a, 19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= rot(*b, 4);
    *b = b.wrapping_add(*a);
}

fn last(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 14));
    *a ^= *c;
    *a = a.wrapping_sub(rot(*c, 11));
    *b ^= *a;
    *b = b.wrapping_sub(rot(*a, 25));
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 16));
    *a ^= *c;
    *a = a.wrapping_sub(rot(*c, 4));
    *b ^= *a;
    *b = b.wrapping_sub(rot(*a, 14));
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 24));
}

fn word(k: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([k[i], k[i + 1], k[i + 2], k[i + 3]])
}

pub(super) fn jenkins(key: &[u8]) -> u32 {
    let seed = 0xdeadbeef_u32.wrapping_add(key.len() as u32);
    let (mut a, mut b, mut c) = (seed, seed, seed);
    let mut k = key;
    while k.len() > 12 {
        a = a.wrapping_add(word(k, 0));
        b = b.wrapping_add(word(k, 4));
        c = c.wrapping_add(word(k, 8));
        mix(&mut a, &mut b, &mut c);
        k = &k[12..];
    }
    let mut tail = [0u8; 12];
    tail[..k.len()].copy_from_slice(k);
    match k.len() {
        0 => return c,
        n => {
            a = a.wrapping_add(u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]));
            if n > 4 {
                b = b.wrapping_add(u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]));
            }
            if n > 8 {
                c = c.wrapping_add(u32::from_le_bytes([tail[8], tail[9], tail[10], tail[11]]));
            }
        }
    }
    last(&mut a, &mut b, &mut c);
    c
}
