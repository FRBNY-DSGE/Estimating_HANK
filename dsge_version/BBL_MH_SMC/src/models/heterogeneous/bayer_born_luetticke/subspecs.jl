"""
`init_subspec!(m::BayerBornLuetticke)`

Initializes a model subspecification by overwriting parameters from
the original model object with new parameter objects. This function is
called from within the model constructor.
"""
function init_subspec!(m::BayerBornLuetticke)
    if subspec(m) == "ss0"
        return
    elseif subspec(m) == "ss1"
        ss1!(m)
    else
        error("This subspec should be a 0")
    end
end

"""
```
ss1!(m::BayerBornLuetticke)
```
sets parameters initialized at zero to eps() so inference
of the Jacobian's sparsity pattern is correct.
"""
function ss1!(m::BayerBornLuetticke)
    m <= parameter(:γ_B_τ, eps(), (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_B_τ: Reaction of tax level to debt",
                   tex_label = "\\gamma_{B, \\tau}")
    m <= parameter(:γ_Y_τ, eps(), (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_Y_τ: Reaction of tax level to output",
                   tex_label = "\\gamma_{Y, \\tau}")
    m <= parameter(:γ_B_P, eps(),(-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_B_P: Reaction of tax level to debt",
                   tex_label = "\\gamma_{B, P}")
    m <= parameter(:γ_Y_P, eps(),(-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_Y_P: Reaction of tax level to output",
                   tex_label = "\\gamma_{Y, P}")
    m <= parameter(:Σ_n, eps(),(-1e3, 1e3), (-1e3, 1e3), ModelConstructors.SquareRoot(),
                   Normal(0., 100.), fixed = false,
                   description = "Σ_n: reaction of income risk to employment status",
                   tex_label = "\\Sigma_{n}")
    m <= parameter(:ρ_R_sh, eps(), (0., 1 - 1e-5), (0., 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_R_ϵ: AR(1) coefficient in the monetary policy shock process.",
                   tex_label = "\\rho_{R, \\epsilon}")
    m <= parameter(:ρ_P_sh, eps(), (0., 1 - 1e-5), (0., 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_P_ϵ: AR(1) coefficient in the tax progressivity shock process.",
                   tex_label = "\\rho_{P, \\epsilon}")
    m <= parameter(:ρ_S_sh, eps(), (0., 1 - 1e-5), (0., 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_S_ϵ: AR(1) coefficient in shock process of the shock in the idiosyncatic income shock process.",
                   tex_label = "\\rho_{S, \\epsilon}")
end
