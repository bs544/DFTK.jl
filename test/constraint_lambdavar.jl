include("run_scf_and_compare.jl")
include("testcases.jl")
using Plots
using HDF5

function finite_diff(x,y)
    x_1 = x[begin:end-1]
    x_2 = x[begin+1:end]
    y_1 = y[begin:end-1]
    y_2 = y[begin+1:end]

    diffs = y_2-y_1
    dxs   = x_2-x_1
    mid_xs= x_1+0.5.*dxs

    grads = diffs./dxs
    return mid_xs,grads
end

function get_scf(model;ecut,kgrid,magnetic_moments,tol=1.0e-10,maxiter=100,α=0.8,fname)
    basis = PlaneWaveBasis(model;Ecut=ecut,kgrid=kgrid)
    ρ0 = guess_density(basis,magnetic_moments)
    scfres = DFTK.self_consistent_field(basis;ρ=ρ0,maxiter,tol,damping=α)
    h5open(fname,"w") do file
        write(file,"rho",ρ0)
    end
    return scfres.ρ
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
                      temperature=0.01, magnetic_moments,symmetries=false)
    model = convert(Model{Float64}, model)
    fname = "./test/constraint_charge_data_iron_pbe_rho.h5"
    if !isfile(fname)
        ρ0 = get_scf(model;ecut=15,kgrid=[3,3,3],magnetic_moments,fname)
    else
        ρ0 = h5open(fname,"r") do file
            read(file,"rho")
        end
    end

    resid_weight = 1.0#0.01
    r_sm_frac = 0.1
    r_cut = 2.0
    spin = 1.5
    charge = 4.736
    idx = 1

    # basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    
    lambdas   = [i for i ∈-0.1:0.01:0.1]
    energies  = []
    grads     = []
    Ns        = []
    gradgrads = []

    constraints     = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_charge=charge,r_cut)]#,DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
    constraint_term = DFTK.DensityMixingConstraint(constraints)
    
    terms = model.term_types
    push!(terms,constraint_term)
    model = Model(model;terms)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    
    # ρ0     = guess_density(basis,magnetic_moments)
    ρ_cons = DFTK.ArrayAndConstraints(ρ0,basis)
    
    diagtol   = 0.003
    nbandsalg = AdaptiveBands(basis.model)
    fermialg  = DFTK.default_fermialg(basis.model)
    
    for lambda in lambdas
        println("lambda: $lambda")
        energy,grad,Hessian = DFTK.EnDerivsFromLagrange([lambda];basis,ρ=ρ0,weights=ρ_cons.weights,tol=diagtol,
                                              ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing,nbandsalg,fermialg,constraints=DFTK.get_constraints(basis))

        N = grad ./ ρ_cons.weights
        N = N[idx] + charge#spin

        push!(energies,energy)
        push!(grads,grad[1])
        push!(Ns,N)
        push!(gradgrads,Hessian[1,1])
    end

    Ws = [energy.total for energy in energies]
    Cons = [energy.energies["DensityMixingConstraint"] for energy in energies]
    Es = [Ws[i]-Cons[i] for i = 1:length(Ws)]

    fd_x,fd_grads = finite_diff(lambdas,Ws)

    p = plot(lambdas,Ws.-Ws[1],label="W")
    plot!(p,lambdas,Es.-Es[1],label="E")
    plot!(p,lambdas,Cons.-Cons[1],label="Cons")
    plot!(p,legend=:bottomleft)
    
    display(p)

    new_plot = plot(lambdas,grads,label="analytic")
    plot!(new_plot,fd_x,fd_grads ,label="finite diff")
    display(new_plot)

    fd_dx,fd_dgrads = finite_diff(lambdas,grads)
    p = plot(lambdas,gradgrads,label="analytic")
    plot!(p,fd_dx,fd_dgrads,label="finite_diff")
    display(p)
    println(gradgrads)



end

run_iron_constrain();