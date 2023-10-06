function precondition(lambda_grads,constraints,detail;ham,basis,ρin,εF,eigenvalues,occupation,ψ,mixing=nothing,n_iter)
  """
  precondition the Lagrange multiplier gradients.
  If detail is "approximate" then the same preconditioning is used as in the density mixing
  If detail is "noninteracting" then the preconditioning is the inverse susceptibility which seems to be the actual second derivative.
  """
  @assert detail ∈ ["approximate","noninteracting"]

  grad_vec = lambdas_2_vector(lambda_grads,constraints)

  #first get (approximate) second derivative for the vector of gradients, can reuse values for approximate hessian
  at_fns_arr = get_4d_at_fns(constraints)
  at_fns = lambdas_2_vector(at_fns_arr,constraints)

  if detail=="approximate"
    χ0_at_fns = [mix_density(mixing,basis,at_fn;ρin,εF,eigenvalues,occupation,ψ,n_iter) for at_fn in at_fns]
  elseif detail=="noninteracting"
    χ0_at_fns = [apply_χ0(ham,ψ,occupation,εF,eigenvalues,at_fn) for at_fn in at_fns]
  end

  Hessian = zeros(Float64,length(at_fns),length(at_fns))
  spin_term = [1 -1;
              -1  1]# copying the hacky spin term thing from the constraint_innerloop.jl routine
  spin_terms = Array{Array{Int,2},2}(undef,size(at_fns_arr)...)
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
                H += sum(at_fns[i][:,:,:,σ1] .* χ0_at_fns[j][:,:,:,σ2])*basis.dvol*spin_terms[i][σ1,σ2]
            end
        end
        Hessian[i,j] = 0.5*H
        Hessian[j,i] = Hessian[i,j]
    end
  end

  if detail=="noninteracting"
    Hessian .*= -1
  end
  #now we have the Hessian (approximate or otherwise), we can precondition by applying it to the gradient and returning it to array form
  #This is just multiplying the gradient by the inverse Hessian (this is the main info I have on this, I should read more on this: https://arxiv.org/pdf/1804.01590.pdf)
  prec_grad_vec = inv(Hessian)*grad_vec
  prec_lambda_grads = vector_2_lambdas(prec_grad_vec,constraints)

  return prec_lambda_grads

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
One of the terms in basis is for constraining the density. These constraints are also important in the mixing of the density,
which is done within a combined struct ArrayAndConstraints
"""
@timing function density_mixed_constrained(
    basis::PlaneWaveBasis{T};
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
    initial_lambda_optimisation=0, # If true, update only the Lagrange multipliers for the specified number of cycles to provide a decent initial guess for the rest of the SCF loop
    lambdas_preconditioning="approximate", # Either approximate or noninteracting. Uses the second derivative calculations from constraint_innerloop.jl and the level of detail found there. 
    response=ResponseOptions(),  # Dummy here, only for AD
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
    info = (; n_iter=0, ρin=ρ)  # Populate info with initial values
    converged = false

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(ρin_cons)
        #ρin_cons: ArrayAndConstraints, contains the density and the constraints
        #ρin     : Density Array
        converged && return ρin_cons  # No more iterations if convergence flagged
        n_iter += 1

        ρin = ρin_cons.arr
        constraints = get_constraints(basis)

        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρin, eigenvalues, εF, 
                                          cons_lambdas=ρin_cons.lambdas, cons_weights=ρin_cons.weights)

        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                 occupation, miniter=1, tol=determine_diagtol(info))
        ψ, eigenvalues, occupation, εF, ρout = nextstate

        resid = residual(ρout,ρin_cons,basis)
        ρout_cons = resid + ρin_cons

        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
                ρin, ρout, α=damping, n_iter, nbandsalg.occupation_threshold,
                nextstate..., diagonalization=[nextstate.diagonalization],
                ρin_cons, ρout_cons)

        # Compute the energy of the new state
        if compute_consistent_energies
            energies = energy_hamiltonian(basis, ψ, occupation;
                                          ρ=ρout, eigenvalues, εF, 
                                          cons_lambdas=ρout_cons.lambdas, cons_weights=ρout_cons.weights).energies
        end
        info = merge(info, (; energies))

        # Apply mixing and pass it the full info as kwargs
        δρ = mix_density(mixing, basis, ρout - ρin; info...)
        resid.lambdas = precondition(resid.lambdas,constraints,lambdas_preconditioning;ham,basis,ρin,εF,eigenvalues,occupation,ψ,mixing,n_iter)
        δρ_cons = ArrayAndConstraints(δρ,resid.lambdas,resid.weights)
        ρnext_cons = ρin_cons .+ T(damping) .* δρ_cons
        info = merge(info, (; ρnext_cons))

        callback(info)
        converged = is_converged(info)
        converged = MPI.bcast(converged, 0, MPI.COMM_WORLD)  # Ensure same converged

        ρnext_cons
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map.
    ρout_cons = ArrayAndConstraints(ρout,basis)
    if !isnothing(initial_lambdas)
      ρout_cons.lambdas = initial_lambdas
    end

    if initial_lambda_optimisation > 0
      #do an initial few steps to update the 
      println("not implemented yet")
    end

    solver(fixpoint_map, ρout_cons, maxiter; tol=eps(T))

    # We do not use the return value of solver but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform a last energy computation
    # to return a correct variational energy
    energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρout_cons.arr, eigenvalues, εF, 
                                       cons_lambdas=ρout_cons.lambdas, cons_weights=ρout_cons.weights)

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout_cons - info.ρin_cons) * sqrt(basis.dvol)

    constraints= get_constraints(basis)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged, nbandsalg.occupation_threshold,
            ρ=ρout, α=damping, eigenvalues, occupation, εF, info.n_bands_converge,
            n_iter, ψ, info.diagonalization, stage=:finalize,
            algorithm="SCF", norm_Δρ,constraints)
    callback(info)
    info
end
