import matplotlib.pyplot as plt
from conspire.math.integrate import BogackiShampine, DormandPrince, Verner8, Verner9

for integrator, name in [
    (BogackiShampine(abs_tol=1e-3, rel_tol=1e-3), "bogacki_shampine"),
    (DormandPrince(abs_tol=1e-6, rel_tol=1e-6), "dormand_prince"),
    (Verner8(abs_tol=1e-11, rel_tol=1e-11), "verner_8"),
    (Verner9(abs_tol=1e-12, rel_tol=1e-12), "verner_9"),
]:
    assert integrator.abs_tol == integrator.rel_tol
    (t, y, _) = integrator.integrate(
        lambda t, y: [y[1], (1 - y[0] ** 2) * y[1] - y[0]], [0, 20], [2, 0]
    )
    plt.plot(t, y, "o-", markerfacecolor="none")

    ax = plt.gca()
    ax.set_title(
        rf"The van der Pol oscillator ($\mu=1$) with tol={integrator.abs_tol:.0e}.",
        fontsize=12,
    )
    ax.set_xlim([0, 20])
    ax.set_ylim([-3, 3])
    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$y(t)$", fontsize=12)

    ax.title.set_color("#000002")
    ax.xaxis.label.set_color("#000002")
    ax.yaxis.label.set_color("#000002")
    ax.tick_params(axis="x", colors="#000002")
    ax.tick_params(axis="y", colors="#000002")

    plt.savefig(f"integrate/explicit/{name}.svg", transparent=True)

    ax.clear()
