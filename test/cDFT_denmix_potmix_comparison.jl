include("run_scf_and_compare.jl")
include("testcases.jl")

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

function run_potmix_cDFT(model,constraints,magnetic_moments,α)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    ρ0 = guess_density(basis,magnetic_moments)
    scfres = DFTK.scf_potential_mixing(basis; tol=1.0e-10,ρ=ρ0,constraints=constraints,maxiter=100,damping=DFTK.FixedDamping(α))
    return scfres
end
    

function run_iron_constrain()
    #TODO: Double check units for this. I think they're off somewhere. Maybe the spin density isn't in μᵦ

    # Produce reference data and guess for this configuration
    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/lda/Fe-q8.hgh"))
    atoms, positions = [Fe,Fe], [zeros(3),0.5.*ones(3)]
    magnetic_moments = [2.0,2.0]
    a = 5.42352
    lattice = a .* [1.0 0.0 0.0;
                    0.0 1.0 0.0;
                    0.0 0.0 1.0]
    model = model_PBE(lattice, atoms, positions;
                      temperature=0.01, magnetic_moments)
    model = convert(Model{Float64}, model)

    idx = 1
    resid_weight = 1.0
    resid_weight_dm = 0.02
    r_sm_frac = 0.1
    r_cut = 2.0
    α = 0.8
    idx = 1


    # basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    
    cons_infos = []
    energies = []
    n_steps = []
    V = nothing
    # ρ0 = guess_density(basis,magnetic_moments)
    for spin in [1.6]
        constraints = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_spin=spin,r_cut)]#,DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
        scfres_pm = run_potmix_cDFT(model,constraints,magnetic_moments,α)
        scfres_dm = run_denmix_cDFT(model,constraints,magnetic_moments,α;resid_weight_dm)

        V_pm = scfres_pm.Vin
        V_dm = DFTK.total_local_potential(scfres_dm.ham)

        λ_pm = scfres_pm.constraints.lambdas[1,2]
        λ_dm = scfres_dm.constraints.lambdas[1,2]

        ρ_pm = scfres_pm.ρ
        ρ_dm = scfres_dm.ρ

        basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
        
        constraints = DFTK.get_constraints(basis)
        println(DFTK.integrate_atomic_functions(spin_density(ρ_pm),constraints,2)[1] - DFTK.integrate_atomic_functions(spin_density(ρ_dm),constraints,2)[1])

        println(scfres_pm.energies.total - scfres_dm.energies.total)

        println(abs(λ_pm)-λ_dm)#, " ", λ_dm)

        # scfres = DFTK.scf_potential_mixing(basis; tol=1.0e-8,ρ=ρ0,V=V,constraints=constraints,maxiter=100,damping=DFTK.FixedDamping(α));
        # push!(cons_infos,scfres.constraints)
        # push!(energies,scfres.energies.total)
        # push!(n_steps,scfres.n_iter)
        # V = DFTK.total_local_potential(scfres.ham)
        # ρ0 = scfres.ρ
    end
    # println("The zero for λ should correspond to the energy minimum")
    # for (i,info) in enumerate(cons_infos)
    #     λ_string  = string(info.lambdas[1,2])
    #     e_string  = string(energies[i]-energies[1])
    #     st_string = string(info.target_values[1,2])
    #     sv_string = string(info.current_values[1,2])
        
    #     λ_string  = λ_string[begin:min(length(λ_string),7)]
    #     e_string  = e_string[begin:min(length(e_string),7)]
    #     st_string = st_string[begin:min(length(st_string),5)]
    #     sv_string = sv_string[begin:min(length(sv_string),5)]
    #     n_step_string = string(n_steps[i])
    #     println("λ: $λ_string \t energies: $e_string  \t nsteps: $n_step_string \t target: $st_string \t value: $sv_string")
    # end
end

run_iron_constrain();