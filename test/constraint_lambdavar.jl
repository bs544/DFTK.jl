include("run_scf_and_compare.jl")
include("testcases.jl")
using Plots

function run_iron_constrain()
    #TODO: Double check units for this. I think they're off somewhere. Maybe the spin density isn't in μᵦ

    # Produce reference data and guess for this configuration
    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/lda/Fe-q8.hgh"))
    atoms, positions = [Fe,Fe], [zeros(3),0.5.*ones(3)]
    magnetic_moments = [2.0,2.0]
    a = 5.42352*0.99
    lattice = a .* [1.0 0.0 0.0;
                    0.0 1.0 0.0;
                    0.0 0.0 1.0]
    model = model_PBE(lattice, atoms, positions;
                      temperature=0.01, magnetic_moments)
    model = convert(Model{Float64}, model)

    resid_weight = 0.01
    r_sm_frac = 0.1
    r_cut = 2.0
    spin = 1.0

    # basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    
    lambdas   = [i for i ∈-0.15:0.05:0.15]
    energies  = []
    grads     = []
    Ns        = []

    constraints     = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_spin=spin,r_cut)]#,DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
    constraint_term = DFTK.DensityMixingConstraint(constraints)
    
    terms = model.term_types
    push!(terms,constraint_term)
    model = Model(model;terms)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    
    ρ0     = guess_density(basis,magnetic_moments)
    ρ_cons = DFTK.ArrayAndConstraints(ρ0,basis)
    
    diagtol   = 0.01
    nbandsalg = AdaptiveBands(basis.model)
    fermialg  = DFTK.default_fermialg(basis.model)
    
    for lambda in lambdas
        energy = DFTK.energy_from_lagrange([lambda];basis,ρ=ρ0,weights=ρ_cons.weights,tol=diagtol,
                                           ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing,nbandsalg,fermialg,constraints=DFTK.get_constraints(basis))
        grad   = DFTK.lagrange_gradient([lambda];basis,ρ=ρ0,weights=ρ_cons.weights,tol=diagtol,
                                           ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing,nbandsalg,fermialg,constraints=DFTK.get_constraints(basis))
        N = grad ./ ρ_cons.weights
        N = N[2] + spin

        push!(energies,energy)
        push!(grads,grad[1])
        push!(Ns,N)
    end

    Ws = [energy.total for energy in energies]
    Cons = [energy.energies["DensityMixingConstraint"] for energy in energies]
    Es = [Ws[i]-Cons[i] for i = 1:length(Ws)]

    p = plot(lambdas,Ws.-Ws[1],label="W")
    plot!(p,lambdas,Es.-Es[1],label="E")
    plot!(p,lambdas,Cons.-Cons[1],label="Cons")
    vline!(p,[-0.111],label="grad=0")
    plot!(p,legend=:bottomleft)
    
    display(p)
    display(plot(lambdas,Ns))

    # display(plot(lambdas,energies))
    # display(plot(lambdas,grads))



end

run_iron_constrain();