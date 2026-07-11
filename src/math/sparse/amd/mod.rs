#[cfg(test)]
mod test;

use super::matrix::CscMatrix;

const NONE: isize = -1;

fn flip(i: isize) -> isize {
    -i - 2
}

fn wclear(mark: isize, lemax: isize, w: &mut [isize], n: usize) -> isize {
    if mark < 2 || mark + lemax < 0 {
        w[..n].iter_mut().for_each(|w_i| {
            if *w_i != 0 {
                *w_i = 1
            }
        });
        2
    } else {
        mark
    }
}

fn tdfs(
    root: isize,
    mut count: usize,
    head: &mut [isize],
    next: &[isize],
    post: &mut [isize],
    stack: &mut [isize],
) -> usize {
    let mut top = 0_isize;
    stack[0] = root;
    while top >= 0 {
        let p = stack[top as usize];
        let i = head[p as usize];
        if i == NONE {
            top -= 1;
            post[count] = p;
            count += 1;
        } else {
            head[p as usize] = next[i as usize];
            top += 1;
            stack[top as usize] = i;
        }
    }
    count
}

impl CscMatrix {
    /// Fill-reducing ordering from approximate minimum degree on the pattern of A + Aᵀ.
    pub fn amd(&self) -> Vec<usize> {
        let n = self.height();
        assert_eq!(n, self.width());
        if n == 0 {
            return Vec::new();
        }
        let transpose = self.transpose();
        let mut cp = vec![0_isize; n + 1];
        let mut ci = Vec::<isize>::new();
        let mut len = vec![0_isize; n + 1];
        (0..n).for_each(|j| {
            cp[j] = ci.len() as isize;
            let mut a = self.column(j).map(|(i, _)| i).peekable();
            let mut b = transpose.column(j).map(|(i, _)| i).peekable();
            loop {
                let i = match (a.peek(), b.peek()) {
                    (Some(&x), Some(&y)) => {
                        if x <= y {
                            if x == y {
                                b.next();
                            }
                            a.next();
                            x
                        } else {
                            b.next();
                            y
                        }
                    }
                    (Some(&x), None) => {
                        a.next();
                        x
                    }
                    (None, Some(&y)) => {
                        b.next();
                        y
                    }
                    (None, None) => break,
                };
                if i != j {
                    ci.push(i as isize);
                }
            }
            len[j] = ci.len() as isize - cp[j];
        });
        let mut cnz = ci.len() as isize;
        let nzmax = cnz + cnz / 5 + 2 * n as isize;
        ci.resize(nzmax as usize, 0);
        let dense = 16
            .max((10.0 * (n as f64).sqrt()) as isize)
            .min(n as isize - 2);
        let mut nv = vec![1_isize; n + 1];
        let mut w = vec![1_isize; n + 1];
        let mut elen = vec![0_isize; n + 1];
        let mut degree = len.clone();
        let mut head = vec![NONE; n + 1];
        let mut next = vec![NONE; n + 1];
        let mut last = vec![NONE; n + 1];
        let mut hhead = vec![NONE; n + 1];
        cp[n] = NONE;
        elen[n] = -2;
        w[n] = 0;
        let mut nel = 0_isize;
        let mut mindeg = 0_usize;
        let mut lemax = 0_isize;
        let mut mark = wclear(0, 0, &mut w, n);
        (0..n).for_each(|i| {
            let d = degree[i];
            if d == 0 {
                elen[i] = -2;
                nel += 1;
                cp[i] = NONE;
                w[i] = 0;
            } else if d > dense {
                nv[i] = 0;
                elen[i] = NONE;
                nel += 1;
                cp[i] = flip(n as isize);
                nv[n] += 1;
            } else {
                if head[d as usize] != NONE {
                    last[head[d as usize] as usize] = i as isize;
                }
                next[i] = head[d as usize];
                head[d as usize] = i as isize;
            }
        });
        while nel < n as isize {
            let mut pivot = NONE;
            while mindeg < n {
                pivot = head[mindeg];
                if pivot != NONE {
                    break;
                }
                mindeg += 1;
            }
            let k = pivot as usize;
            if next[k] != NONE {
                last[next[k] as usize] = NONE;
            }
            head[mindeg] = next[k];
            let elenk = elen[k];
            let mut nvk = nv[k];
            nel += nvk;
            if elenk > 0 && cnz + mindeg as isize >= nzmax {
                (0..n).for_each(|j| {
                    let p = cp[j];
                    if p >= 0 {
                        cp[j] = ci[p as usize];
                        ci[p as usize] = flip(j as isize);
                    }
                });
                let mut q = 0_usize;
                let mut p = 0_usize;
                while (p as isize) < cnz {
                    let j = flip(ci[p]);
                    p += 1;
                    if j >= 0 {
                        let j = j as usize;
                        ci[q] = cp[j];
                        cp[j] = q as isize;
                        q += 1;
                        (0..len[j] - 1).for_each(|_| {
                            ci[q] = ci[p];
                            q += 1;
                            p += 1;
                        });
                    }
                }
                cnz = q as isize;
            }
            let mut dk = 0_isize;
            nv[k] = -nvk;
            let mut p = cp[k];
            let pk1 = if elenk == 0 { p } else { cnz };
            let mut pk2 = pk1;
            for k1 in 1..=(elenk + 1) {
                let (e, mut pj, ln) = if k1 > elenk {
                    (k, p, len[k] - elenk)
                } else {
                    let e = ci[p as usize] as usize;
                    p += 1;
                    (e, cp[e], len[e])
                };
                (0..ln).for_each(|_| {
                    let i = ci[pj as usize] as usize;
                    pj += 1;
                    let nvi = nv[i];
                    if nvi > 0 {
                        dk += nvi;
                        nv[i] = -nvi;
                        ci[pk2 as usize] = i as isize;
                        pk2 += 1;
                        if next[i] != NONE {
                            last[next[i] as usize] = last[i];
                        }
                        if last[i] != NONE {
                            next[last[i] as usize] = next[i];
                        } else {
                            head[degree[i] as usize] = next[i];
                        }
                    }
                });
                if e != k {
                    cp[e] = flip(k as isize);
                    w[e] = 0;
                }
            }
            if elenk != 0 {
                cnz = pk2;
            }
            degree[k] = dk;
            cp[k] = pk1;
            len[k] = pk2 - pk1;
            elen[k] = -2;
            mark = wclear(mark, lemax, &mut w, n);
            (pk1..pk2).for_each(|pk| {
                let i = ci[pk as usize] as usize;
                let eln = elen[i];
                if eln > 0 {
                    let nvi = -nv[i];
                    let wnvi = mark - nvi;
                    (cp[i]..cp[i] + eln).for_each(|p| {
                        let e = ci[p as usize] as usize;
                        if w[e] >= mark {
                            w[e] -= nvi;
                        } else if w[e] != 0 {
                            w[e] = degree[e] + wnvi;
                        }
                    });
                }
            });
            (pk1..pk2).for_each(|pk| {
                let i = ci[pk as usize] as usize;
                let p1 = cp[i];
                let p2 = p1 + elen[i];
                let mut pn = p1;
                let mut hash = 0_usize;
                let mut d = 0_isize;
                (p1..p2).for_each(|p| {
                    let e = ci[p as usize] as usize;
                    if w[e] != 0 {
                        let dext = w[e] - mark;
                        if dext > 0 {
                            d += dext;
                            ci[pn as usize] = e as isize;
                            pn += 1;
                            hash += e;
                        } else {
                            cp[e] = flip(k as isize);
                            w[e] = 0;
                        }
                    }
                });
                elen[i] = pn - p1 + 1;
                let p3 = pn;
                let p4 = p1 + len[i];
                (p2..p4).for_each(|p| {
                    let j = ci[p as usize] as usize;
                    let nvj = nv[j];
                    if nvj > 0 {
                        d += nvj;
                        ci[pn as usize] = j as isize;
                        pn += 1;
                        hash += j;
                    }
                });
                if d == 0 {
                    cp[i] = flip(k as isize);
                    let nvi = -nv[i];
                    dk -= nvi;
                    nvk += nvi;
                    nel += nvi;
                    nv[i] = 0;
                    elen[i] = NONE;
                } else {
                    degree[i] = degree[i].min(d);
                    ci[pn as usize] = ci[p3 as usize];
                    ci[p3 as usize] = ci[p1 as usize];
                    ci[p1 as usize] = k as isize;
                    len[i] = pn - p1 + 1;
                    let hash = hash % n;
                    next[i] = hhead[hash];
                    hhead[hash] = i as isize;
                    last[i] = hash as isize;
                }
            });
            degree[k] = dk;
            lemax = lemax.max(dk);
            mark = wclear(mark + lemax, lemax, &mut w, n);
            (pk1..pk2).for_each(|pk| {
                let i = ci[pk as usize] as usize;
                if nv[i] < 0 {
                    let hash = last[i] as usize;
                    let mut i = hhead[hash];
                    hhead[hash] = NONE;
                    while i != NONE && next[i as usize] != NONE {
                        let iu = i as usize;
                        let ln = len[iu];
                        let eln = elen[iu];
                        (cp[iu] + 1..cp[iu] + ln).for_each(|p| w[ci[p as usize] as usize] = mark);
                        let mut jlast = iu;
                        let mut j = next[iu];
                        while j != NONE {
                            let ju = j as usize;
                            let mut ok = len[ju] == ln && elen[ju] == eln;
                            let mut p = cp[ju] + 1;
                            while ok && p < cp[ju] + ln {
                                if w[ci[p as usize] as usize] != mark {
                                    ok = false;
                                }
                                p += 1;
                            }
                            if ok {
                                cp[ju] = flip(i);
                                nv[iu] += nv[ju];
                                nv[ju] = 0;
                                elen[ju] = NONE;
                                j = next[ju];
                                next[jlast] = j;
                            } else {
                                jlast = ju;
                                j = next[ju];
                            }
                        }
                        i = next[iu];
                        mark += 1;
                    }
                }
            });
            let mut p = pk1;
            (pk1..pk2).for_each(|pk| {
                let i = ci[pk as usize] as usize;
                let nvi = -nv[i];
                if nvi > 0 {
                    nv[i] = nvi;
                    let d = (degree[i] + dk - nvi).min(n as isize - nel - nvi);
                    if head[d as usize] != NONE {
                        last[head[d as usize] as usize] = i as isize;
                    }
                    next[i] = head[d as usize];
                    last[i] = NONE;
                    head[d as usize] = i as isize;
                    mindeg = mindeg.min(d as usize);
                    degree[i] = d;
                    ci[p as usize] = i as isize;
                    p += 1;
                }
            });
            nv[k] = nvk;
            len[k] = p - pk1;
            if len[k] == 0 {
                cp[k] = NONE;
                w[k] = 0;
            }
            if elenk != 0 {
                cnz = p;
            }
        }
        (0..=n).for_each(|i| cp[i] = flip(cp[i]));
        head.iter_mut().for_each(|head_i| *head_i = NONE);
        (0..n).rev().for_each(|j| {
            if nv[j] <= 0 {
                next[j] = head[cp[j] as usize];
                head[cp[j] as usize] = j as isize;
            }
        });
        (0..=n).rev().for_each(|e| {
            if nv[e] > 0 && cp[e] != NONE {
                next[e] = head[cp[e] as usize];
                head[cp[e] as usize] = e as isize;
            }
        });
        let mut post = vec![0_isize; n + 1];
        let mut stack = vec![0_isize; n + 1];
        let mut count = 0_usize;
        (0..=n).for_each(|i| {
            if cp[i] == NONE {
                count = tdfs(i as isize, count, &mut head, &next, &mut post, &mut stack);
            }
        });
        post.into_iter()
            .filter(|&i| i != n as isize)
            .map(|i| i as usize)
            .collect()
    }
}
