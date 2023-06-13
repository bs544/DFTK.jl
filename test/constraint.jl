include("run_scf_and_compare.jl")
include("testcases.jl")

function run_iron_constrain()
    #TODO: Double check units for this. I think they're off somewhere. Maybe the spin density isn't in μᵦ

    # Produce reference data and guess for this configuration
    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/lda/Fe-q8.hgh"))
    atoms, positions = [Fe], [zeros(3)]
    magnetic_moments = [2.0]
    model = model_PBE(iron_bcc.lattice.*1.10, iron_bcc.atoms, iron_bcc.positions;
                      temperature=0.01, magnetic_moments)
    model = convert(Model{Float64}, model)

    idx = 1
    spin_target = 1.5
    resid_weight = 1.5
    rsm = 0.05*Fe.psp.rloc
    α = 0.8

    constraints = [DFTK.Constraint(model,idx,spin_target,resid_weight,rsm)]

    basis = PlaneWaveBasis(model; Ecut=20, fft_size=[20, 20, 20],
                           kgrid=[4, 4, 4], kshift=[1/2, 1/2, 1/2])
    ρ0 = guess_density(basis,magnetic_moments)
    # scfres  = DFTK.scf_potential_mixing(basis; tol=1.0e-8,ρ=ρ0);
    scfres2 = DFTK.scf_potential_mixing(basis; tol=1.0e-12,ρ=ρ0,constraints=constraints,maxiter=100,damping=DFTK.FixedDamping(α)); 
end


run_iron_constrain();