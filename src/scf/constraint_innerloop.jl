"""
This is to provide an optimised set of Lagrange multipliers given a trial density. 
This is done by defining functions that give E and ∂E/∂λᵢ and plugging that into Optim.
The point of this is to provide a comparison between inner loop methods and the density mixing cDFT implementation
"""

function lambdas_2_vector(lambdas,constraints::Constraints)
    """
    For the purposes of minimisation, the lambdas array needs to be a vector of only the elements that are being constrained
    """
    T = typeof(lambdas[1])
    T = T<:Int ? Number : T #generalise if T is an integer, since this can happen if lambdas[1] is 0
    λ = Vector{T}(undef,sum(constraints.is_constrained))
    idx = 1
    for i in eachindex(lambdas)
        if constraints.is_constrained[i]==1
            λ[idx] = lambdas[i]
            idx +=1
        end
    end
    @assert idx-1 == sum(constraints.is_constrained)
    return λ
end

function vector_2_lambdas(λ,constraints::Constraints)
    """
    turn the vector back to the array
    """
    lambdas = zeros(Number,size(constraints.is_constrained))
    idx = 1
    for i in eachindex(lambdas)
        if constraints.is_constrained[i]==1
            lambdas[i]= λ[idx]
            idx += 1
        end
    end
    return lambdas
end

function update_constraints!(basis::PlaneWaveBasis,lambdas)
    for term in basis.terms
        if :constraints in fieldnames(typeof(term))
            term.constraints.lambdas = lambdas
        end
    end
end

function W_λ(ham,basis,ρout,ρin,ψ,occupation,eigenvalues,εF,cons_lambdas)
    """
    Get the energy which is optimised by the Hamiltonian diagonalisation. This has the potential generated by ρin, but uses the density ρout, and the output wavefunctions
    Ultimately this involves finding ∫ V_Hxc[ρin](x)*ρout(x)dx and using this to replace E_out.Hartree and E_out.Xc:
    E_out - E_out.H -E_out.XC + ∫(v_H[ρin](x)+v_xc[ρin](x))(ρout(x)-0.5*ρin(x))dx
    """
    V_loc = total_local_potential(ham)
    V_H =  zeros(Float64,size(V_loc)...)
    V_XC = zeros(Float64,size(V_loc)...)

    spin_idxs = [first(krange_spin(basis,σ)) for σ = 1:basis.model.n_spin_components]

    spin_up_block = ham.blocks[spin_idxs[1]]
    spin_dn_block = length(spin_idxs)==2 ? ham.blocks[spin_idxs[2]] : nothing

    for i = 1:length(ham.basis.terms)
        up_op = spin_up_block.operators[i]
        dn_op = isnothing(spin_dn_block) ? nothing : spin_dn_block.operators[i]

        term = ham.basis.terms[i]
        t_type = typeof(term)
        if t_type <: TermHartree 
            pot_up = up_op.potential 
            pot_dn = isnothing(dn_op) ? nothing : dn_op.potential
            V_H = isnothing(pot_dn) ? pot_up : cat([pot_up,pot_dn]... ,dims=4)
        elseif t_type <: TermXc 
            pot_up = up_op.potential 
            pot_dn = isnothing(dn_op) ? nothing : dn_op.potential
            V_XC = isnothing(pot_dn) ? pot_up : cat([pot_up,pot_dn]... ,dims=4)
        end
    end

    V_loc = V_XC + V_H
    E_HXC_from_pot = sum((ρout.-0.5.*ρin).*V_loc)*basis.dvol
    E_out = energy_hamiltonian(basis,ψ,occupation;ρ=ρout,eigenvalues,εF,cons_lambdas).energies #this will have most of the terms we want

    E = E_out.total - E_out.Hartree - E_out.Xc + E_HXC_from_pot
    
    return E
end

function dielectric_operator(δρ,basis,ρ,ham,ψ,occupation,εF,eigenvalues)
    δV = apply_kernel(basis,δρ;ρ)
    χ0δV = apply_χ0(ham,ψ,occupation,εF,eigenvalues,δV)
    return δρ - χ0δV
end

function second_deriv_wrt_lagrange(λ,constraints,interacting=false;basis,ρ,ham,ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing,tol=nothing,nbandsalg=nothing,fermialg=nothing,eigensolver=lobpcg_hyper)

    lambdas = vector_2_lambdas(λ,constraints)
    if nothing in [ψ,occupation,εF,eigenvalues]
        #assume ρ is ρin
        #need to generate a hamiltonian to diagonalise
        if isnothing(ham)
            ham = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas).ham
        end
        ψ, eigenvalues, occupation, εF, ρ = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                                         occupation, miniter=1, tol)
    end

    ε(arr) = dielectric_operator(arr,basis,ρ,ham,ψ,occupation,εF,eigenvalues)

    at_fn_arrs = get_4d_at_fns(constraints)
    at_fns = lambdas_2_vector(at_fn_arrs,constraints)
    χ0_at_fns = [apply_χ0(ham,ψ,occupation,εF,eigenvalues,at_fn) for at_fn in at_fns]
    if interacting
        # apply the inverse of ε(arr) to χ0_at_fn(=∫χ0(x,x')wᵢ(x')dx') to get χwᵢ
        inv_εs = [linsolve(arr->ε(arr),χ0_at_fn,verbosity=3)[1] for χ0_at_fn in χ0_at_fns]
    else
        #just stick with χ0wᵢ
        inv_εs = χ0_at_fns
    end

    Hessian = zeros(Float64,length(at_fns),length(at_fns))

    #this whole spin term thing is *super* hacky and may prevent this stuff from being extended beyond just spins and charges
    spin_term = [1 -1;
                -1  1]
    spin_terms = Array{Array{Int,2},2}(undef,size(at_fn_arrs)...)
    for i in CartesianIndices(spin_terms)
        arr = i.I[2]==2 ? spin_term : ones(Int,2,2)
        spin_terms[i] = arr
    end
    spin_terms = lambdas_2_vector(spin_terms,constraints)
    for i = 1:length(at_fns)
        for j = i:length(at_fns)
            H = 0.0
            for σ1 = 1:2
                for σ2 = 1:2 
                    H += sum(at_fns[i][:,:,:,σ1] .* inv_εs[j][:,:,:,σ2])*basis.dvol*spin_terms[i][σ1,σ2]
                end
            end
            Hessian[i,j] = 0.5*H
            Hessian[j,i] = Hessian[i,j]
        end
    end
    return Hessian
end

function EnDerivsFromLagrange(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,eigensolver=lobpcg_hyper,nbandsalg,fermialg,tol=0.003,constraints,do_hessian=true,do_grad=true)
    """
    Convert the λ vector to the array used by the constraints
    Generate the Hamiltonian and diagonalise it
    Use ρin, ρout and (ψ,occ) to generate the Energy and its first and second derivatives
    dE/dλᵢ = Nᵢ-Nᵢᵗ
    d²E/dλᵢdλⱼ = dNᵢ/dλⱼ = 1/2 ∑ₛₛ₋ ∫ wᵢˢ(x) χˢˢ⁻(x,x') wᵢˢ⁻(x') dx dx'
    Do all in one function to minimise the number of Hamiltonian diagonalisations 
    Have the option of skipping the hessian calculation since it's pretty costly
    """
    lambdas = vector_2_lambdas(λ,constraints)
    ham = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas, cons_weights=weights).ham
    ψ, eigenvalues, occupation, εF, ρout = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                                        occupation, miniter=1, tol)
    E = W_λ(ham,basis,ρout,ρ,ψ,occupation,eigenvalues,εF,lambdas)

    deriv_array = residual(ρout,ArrayAndConstraints(ρ,lambdas,weights),basis).lambdas
    deriv_array ./= weights
    deriv_array = do_grad ? lambdas_2_vector(deriv_array,constraints) : nothing
    
    Hessian = do_hessian ? second_deriv_wrt_lagrange(λ,constraints;basis,ρ=ρout,ham,ψ,occupation,εF,eigenvalues) : nothing

    return E, deriv_array, Hessian
end

function innerloop!(ρ,basis,nbandsalg,fermialg,diagtol,λ_tol,max_cons_iter,n_Ham_diags;ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing)
    """
    Find the value for the Lagrange Multipliers that satisfies the constraints kept in basis.
    Beyond the expected inputs needed to form and diagonalise the Hamiltonian, the two other inputs are:
        λ_tol         : the tolerance in the values of the λ values below which the optimisation finishes
        max_cons_iter : the maximum number of iterations of this constraining optimisation. I don't know what this should be set to at the moment, surely 10 would do?
    """
    constraints = get_constraints(basis)
    lambdas = constraints.lambdas
    weights = ones(Float64,size(lambdas)...)

    function fgh!(F,G,H,λ)
        do_hessian = !isnothing(H)
        do_grad = !isnothing(G)
        E,_G,_H = EnDerivsFromLagrange(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,nbandsalg,fermialg,tol=diagtol,constraints,do_hessian,do_grad)
        if do_grad
            G[:] = -_G[:]
        end
        if do_hessian
            H[:,:] = -_H[:,:]
        end
        isnothing(F) || return -E #have to make all of these negative since you're maximising the energy and Optim minimises functions
        return nothing
    end
    λ = lambdas_2_vector(lambdas,constraints)
    λ = zeros(Float64,size(λ))
    optim_results = Optim.optimize(Optim.only_fgh!(fgh!),λ,method=Newton();
                                   store_trace=true,extended_trace=true,
                                   iterations=max_cons_iter,show_trace=false,inplace=true,x_tol=λ_tol)
    new_lambdas= Optim.x_trace(optim_results)[end]

    # println(optim_results)
    # println(Optim.x_trace(optim_results))
    new_lambdas = vector_2_lambdas(new_lambdas,constraints)
    update_constraints!(basis,new_lambdas)
    push!(n_Ham_diags,Optim.f_calls(optim_results))#use this as a proxy for the number of Hamiltonian diagonalisations needed for λ convergence
end

@doc raw"""
    density_mixed_constrained(basis; [tol, mixing, damping, ρ, ψ])

Solve the Kohn-Sham equations with a density-based SCF algorithm using damped, preconditioned
iterations where ``ρ_\text{next} = α P^{-1} (ρ_\text{out} - ρ_\text{in})``.

Overview of parameters:
- `ρ`:   Initial density
- `ψ`:   Initial orbitals
- `tol`: Tolerance for the density change (``\|ρ_\text{out} - ρ_\text{in}\|``)
  to flag convergence. Default is `1e-6`.
- `is_converged`: Convergence control callback. Typical objects passed here are
  `DFTK.ScfConvergenceDensity(tol)` (the default), `DFTK.ScfConvergenceEnergy(tol)`
  or `DFTK.ScfConvergenceForce(tol)`.
- `maxiter`: Maximal number of SCF iterations
- `mixing`: Mixing method, which determines the preconditioner ``P^{-1}`` in the above equation.
  Typical mixings are [`LdosMixing`](@ref), [`KerkerMixing`](@ref), [`SimpleMixing`](@ref)
  or [`DielectricMixing`](@ref). Default is `LdosMixing()`
- `damping`: Damping parameter ``α`` in the above equation. Default is `0.8`.
- `nbandsalg`: By default DFTK uses `nbandsalg=AdaptiveBands(model)`, which adaptively determines
  the number of bands to compute. If you want to influence this algorithm or use a predefined
  number of bands in each SCF step, pass a [`FixedBands`](@ref) or [`AdaptiveBands`](@ref).
- `callback`: Function called at each SCF iteration. Usually takes care of printing the
  intermediate state.

My modification:
One of the terms in basis is for constraining the density. 
The constraining lagrange multipliers are found in an inner loop using the Newton method as implemented in Optim.jl
"""
@timing function scf_constrained_innerloop(
    basis::PlaneWaveBasis{T}; #this should have the constraints already applied
    ρ=guess_density(basis),
    ψ=nothing,
    tol=1e-6,
    is_converged=ScfConvergenceDensity(tol),
    maxiter=100,
    mixing=LdosMixing(),
    damping=0.8,
    solver=scf_anderson_solver(),
    eigensolver=lobpcg_hyper,
    determine_diagtol=ScfDiagtol(),
    nbandsalg::NbandsAlgorithm=AdaptiveBands(basis.model),
    fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model),
    callback=ScfDefaultCallback(; show_damping=false),
    compute_consistent_energies=true,
    initial_lambdas=nothing,
    response=ResponseOptions(),  # Dummy here, only for AD
    λ_tol= 1e-5 # tolerance for constraining lagrange multipliers in inner loop
) where {T}
    # All these variables will get updated by fixpoint_map
    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
    end
    occupation = nothing
    eigenvalues = nothing
    ρout = ρ
    εF = nothing
    n_iter = 0
    energies = nothing
    ham = nothing
    info = (; n_iter=0, ρin=ρ,n_Ham_diags=[])  # Populate info with initial values
    converged = false

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(ρin)
        #ρin_cons: ArrayAndConstraints, contains the density and the constraints
        #ρin     : Density Array
        # for this innerloop stuff, the constraining information is kept in the basis.
        converged && return ρin  # No more iterations if convergence flagged
        n_iter += 1

        #find the Lagrange multipliers that ensure that ρout obeys the constraints
        #the constraints in the basis term are updated with these values
        n_Ham_diags = info.n_Ham_diags
        diagtol = determine_diagtol(info)
        max_cons_iter = 20
        innerloop!(ρin,basis,nbandsalg,fermialg,diagtol,λ_tol,max_cons_iter,n_Ham_diags;ψ,occupation,εF,eigenvalues)
        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρin, eigenvalues, εF)

        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                 occupation, miniter=1, tol=determine_diagtol(info))
        ψ, eigenvalues, occupation, εF, ρout = nextstate

        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
                ρin, ρout, α=damping, n_iter, nbandsalg.occupation_threshold,
                nextstate..., diagonalization=[nextstate.diagonalization],
                n_Ham_diags)

        # Compute the energy of the new state
        if compute_consistent_energies
            energies = energy_hamiltonian(basis, ψ, occupation;
                                          ρ=ρout, eigenvalues, εF).energies
        end
        info = merge(info, (; energies))

        # Apply mixing and pass it the full info as kwargs
        δρ = mix_density(mixing, basis, ρout - ρin; info...)
        ρnext = ρin .+ T(damping) .* δρ
        info = merge(info, (; ρnext))

        callback(info)
        converged = is_converged(info)
        converged = MPI.bcast(converged, 0, MPI.COMM_WORLD)  # Ensure same converged

        ρnext
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map.
    if !isnothing(initial_lambdas)
      update_constraints!(basis,initial_lambdas)
    end
    solver(fixpoint_map, ρ, maxiter; tol=eps(T))

    # We do not use the return value of solver but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform a last energy computation
    # to return a correct variational energy
    energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=info.ρout, eigenvalues, εF)

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout - info.ρin) * sqrt(basis.dvol)
    println(info.n_Ham_diags)

    constraints= get_constraints(basis)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged, nbandsalg.occupation_threshold,
            ρ=ρout, α=damping, eigenvalues, occupation, εF, info.n_bands_converge,
            n_iter, ψ, info.diagonalization, stage=:finalize,
            algorithm="SCF", norm_Δρ,constraints,info.n_Ham_diags)
    callback(info)
    info
end
