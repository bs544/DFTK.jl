include("run_scf_and_compare.jl")
include("testcases.jl")
using Plots
using HDF5

function finite_diff(x,y)
    x_1 = x[begin:end-1]
    x_2 = x[begin+1:end]
    y_1 = y[begin:end-1]
    y_2 = y[begin+1:end]

    dys = y_2-y_1
    dxs   = x_2-x_1
    mid_xs= x_1+0.5.*dxs

    grads = dys./dxs
    return mid_xs,grads
end

function central_diff(x,y)
    δx = x[2]-x[1]
    y_1 = y[begin:end-2]
    y_2 = y[begin+2:end]

    δy = y_2-y_1

    grads = δy./(2*δx)
    return x[begin+1:end-1],grads
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
                      temperature=0.01, symmetries=false,magnetic_moments)
    model = convert(Model{Float64}, model)
    # magnetic_moments = [0.1,0.0]
    # model = model_LDA(silicon.lattice,silicon.atoms,silicon.positions; temperature=0.001,smearing=Smearing.Gaussian(),symmetries=false,magnetic_moments)
    # model = convert(Model{Float64},model)


    resid_weight = 0.01
    r_sm_frac = 0.1
    r_cut = 2.0
    spin = 1.5
    charge = 4.736
    spin_cons = true
    if spin_cons
        idx = 2
        target = spin
        fname = "./test/constraint_spin_data_iron_pbe_rho.h5"
        constraints     = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_spin=target,r_cut)]#,DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
    else
        idx = 1
        target = charge
        fname = "./test/constraint_charge_data_iron_pbe_rho.h5"
        constraints     = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_charge=target,r_cut)]#,DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
    end

    if !isfile(fname)
        ρ0 = get_scf(model;ecut=15,kgrid=[3,3,3],magnetic_moments,fname)
    else
        ρ0 = h5open(fname,"r") do file
            read(file,"rho")
        end
    end
    
    lambdas   = [i for i ∈0.01:0.0025:0.03]
    energies  = []
    grads     = []
    Es        = []
    Ns        = []
    cons_es   = []
    kin_es    = []
    ent_es    = []
    sum_es    = []
    loc_es    = []

    constraint_term = DFTK.DensityMixingConstraint(constraints)
    
    terms = model.term_types
    new_terms = []
    xc_idx = 0
    for (i,term) in enumerate(terms)
        if true #!(typeof(term)<:Xc || typeof(term)<:Hartree || typeof(term) <: Ewald || typeof(term) <: PspCorrection || typeof(term) <: AtomicNonlocal || typeof(term) <: AtomicLocal)
            push!(new_terms,term)
        end
        if typeof(term) <: Xc
            xc_idx = i
        end
    end
    push!(new_terms,constraint_term)
    model = Model(model;terms=new_terms)
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=[2,2,2])
    
    ρ0     = guess_density(basis,magnetic_moments)
    ρ_cons = DFTK.ArrayAndConstraints(ρ0,basis)
    
    diagtol   = 0.003
    nbandsalg = AdaptiveBands(basis.model)
    fermialg  = DFTK.default_fermialg(basis.model)
    
    for lambda in lambdas
        println("lambda: $lambda")
        energy,grad, ham_out,ρout,E = DFTK.EnGradFromLagrange([lambda];basis,ρ=ρ0,weights=ρ_cons.weights,tol=diagtol,
                                              ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing,nbandsalg,fermialg,constraints=DFTK.get_constraints(basis))
        N = grad[1]
        N = N + target

        constraints=DFTK.get_constraints(basis)
        E_cons,V_op = DFTK.ene_ops(DFTK.TermDensityMixingConstraint(constraints),basis,nothing,nothing;ρ = ρout,cons_lambdas=constraints.lambdas)
        spin_indices = [first(DFTK.krange_spin(basis,σ)) for σ = 1:basis.model.n_spin_components]
        _,V_xc = DFTK.ene_ops(basis.terms[xc_idx],basis,nothing,nothing;ρ=ρout)
        V_xc_up = V_xc[spin_indices[1]].potential
        # V_xc_dn = V_xc[spin_indices[2]].potential
        println(V_xc_up[1])#,"\t",V_xc_dn[1])
        # V_up = V_op[spin_indices[1]].potential
        # V_dn = V_op[spin_indices[2]].potential
        # V_ham = DFTK.total_local_potential(ham_out)
        # println(maximum(abs.(V_up)))#,"\t",maximum(abs.(V_up[:,:,:,1])),"\t",maximum(abs.(V_up[:,:,:,2])))
        # println(maximum(abs.(V_dn)))#,"\t",maximum(abs.(V_dn[:,:,:,1])),"\t",maximum(abs.(V_dn[:,:,:,2])))
        # println(maximum(abs.(V_ham)),"\t",maximum(abs.(V_ham[:,:,:,2])))

        n = DFTK.integrate_atomic_functions(ρout[:,:,:,1],constraints,1)[1]
        m = DFTK.integrate_atomic_functions(total_density(ρout),constraints,1)[1]
        println(m/n)
        cons_e = lambda * (m-constraints.target_values[1,1])
        println(cons_e/E_cons)

        push!(cons_es,E_cons)
        push!(kin_es,E.Kinetic)
        push!(ent_es,E.Entropy)
        # push!(loc_es,E.AtomicLocal)
        push!(sum_es,E.Entropy+E_cons)#+E.Kinetic)

        # energy[1] += cons_e - E_cons

        push!(energies, energy)
        push!(grads,grad[1])
        push!(Ns,n)
    end
    # println(energies)
    labels = energies[1][end]
    Es = [[e[i] for e in energies] for i = 1:length(labels)]


    plot1 = plot(lambdas,grads,label="N-Nᵗ")    
    plot2 = plot()
    plot3 = plot()#lambdas,Ns,label="N")

    sum_grads = zeros(Float64,length(lambdas)-2)
    fd_e = nothing
    for (e_test,label) in zip(Es,labels)
        fd_x,fd_e = central_diff(lambdas,e_test)
        if label != "∑ᵢ εᵢ"
            plot!(plot1,fd_x,fd_e,label=label)
        else
            plot!(plot3,fd_x,fd_e.-Ns[begin+1:end-1],label="$label - N")
        end
        println(label)
        # plot!(plot2,fd_x,fd_e-grads[begin+1:end-1],label=label)
    end
    fd_x,fd_e_cons = central_diff(lambdas,cons_es)
    # plot!(plot1,fd_x,fd_e_cons,label="Cons E")
    fd_x,fd_e_kin = central_diff(lambdas,kin_es)
    plot!(plot2,fd_x,fd_e_kin,label="Kin E")
    fd_x,fd_e_ent = central_diff(lambdas,ent_es)
    plot!(plot2,fd_x,fd_e_ent,label="Ent E")
    # fd_x,fd_e_loc = central_diff(lambdas,loc_es)
    # plot!(plot2,fd_x,fd_e_loc,label="Loc E")


    fd_x,fd_N = central_diff(lambdas,Ns)
    plot!(plot2,fd_x,fd_N.*fd_x,label="λ dN/dλ")
    sum_grads .+=  fd_e_ent + fd_N.*fd_x + fd_e_kin #+ fd_e_loc 

    plot!(plot2,fd_x,sum_grads,label="Sum")
    

    plot!(plot1,legend=:outerbottom)
    # plot!(plot2,legend=:outerbottom)
    display(plot1)
    # display(plot2)
    # display(plot3)

    


end

run_iron_constrain();