include("run_scf_and_compare.jl")
include("testcases.jl")

function update_constraints!(basis::PlaneWaveBasis,lambdas)
    for term in basis.terms
        if :constraints in fieldnames(typeof(term))
            term.constraints.lambdas = lambdas
        end
    end
end

function initial_lambda(model,constraints,magnetic_moments,initial_optimize)
    constraint_term = DFTK.DensityMixingConstraint(constraints)
    terms = model.term_types
    push!(terms,constraint_term)
    tmp_model = Model(model;terms)
    basis = PlaneWaveBasis(tmp_model; Ecut=15, kgrid=[3,3,3])
    if initial_optimize
        ρ0 = guess_density(basis,magnetic_moments)
        ρ_cons = DFTK.ArrayAndConstraints(ρ0,basis)
        diagtol = 0.01
        λ_tol = 1e-3
        max_cons_iter = 10
        lambdas = DFTK.innerloop(ρ_cons,basis,AdaptiveBands(basis.model),DFTK.default_fermialg(basis.model),diagtol,λ_tol,max_cons_iter)
        update_constraints!(basis,lambdas)
    end
    return basis
end

function run_denmix_cDFT(model,constraints,magnetic_moments,α;resid_weight_dm=nothing)
    if !isnothing(resid_weight_dm)
        for (i,constraint) in enumerate(constraints)
            constraints[i] = DFTK.Constraint(constraint.atom_pos,constraint.atom_idx,constraint.spin,constraint.charge,
                                         constraint.r_sm,constraint.r_cut,constraint.target_spin,constraint.target_charge,resid_weight_dm)
        end
    end
    constraint_term = DFTK.DensityMixingConstraint(constraints)
    terms = model.term_types
    push!(terms,constraint_term)
    tmp_model = Model(model;terms)
    basis = PlaneWaveBasis(tmp_model; Ecut=15, kgrid=[3,3,3])
    ρ0 = guess_density(basis,magnetic_moments)
    scfres = DFTK.density_mixed_constrained(basis; tol=1.0e-10,ρ=ρ0,maxiter=100,damping=α)
    return scfres
end

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

    idx = 1
    resid_weight = 0.01
    r_sm_frac = 0.1
    r_cut = 2.0
    α = 0.8
    idx = 1
    initial_optimize = false

    # basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    
    cons_infos = []
    energies = []
    n_steps = []
    V = nothing
    # ρ0 = guess_density(basis,magnetic_moments)
    for spin in [1.6]
        constraints = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_spin=spin,r_cut),DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
        basis = initial_lambda(model,constraints,magnetic_moments,initial_optimize)
        ρ0 = guess_density(basis,magnetic_moments)
        scfres = DFTK.density_mixed_constrained(basis; tol=1.0e-10,ρ=ρ0,maxiter=100,damping=α)




    end

end

run_iron_constrain();