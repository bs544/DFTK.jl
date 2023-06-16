"""
functions needed to constrain some target properties. e.g. atomic spin

The potential mixing constrained DFT method only requires a change to the residual before preconditioning

The density mixing constrained DFT method is a bit more involved.

NOTE: Charge constraints are harder to converge. Try smaller mixing parameters and smaller weights for the constrained part of the residual
"""

mutable struct Constraint
    """
    This is supposed to define the constraint on the spin and/or charge around an atom
    Just use one of these per atom
    """
    atom_pos          :: Vector{Float64}
    atom_idx          :: Int     #index of atom as specified in model
    spin              :: Bool
    charge            :: Bool
    r_sm              :: Float64 #cutoff radius for atomic function
    r_cut             :: Float64 #smearing width for atomic function
    target_spin       :: Float64
    target_charge     :: Float64
    current_spin      :: Float64
    current_charge    :: Float64
    cons_resid_weight :: Float64
    λ_charge          :: Float64 #Lagrange multiplier for constraint
    λ_spin            :: Float64
    unit_cell_volume  :: Float64 #sometimes useful to have, otherwise set to zero
end

function Constraint(model::Model,idx::Int,cons_resid_weight::Float64=1.0,r_sm_frac::Float64=0.05;target_spin=nothing,target_charge=nothing)::Constraint
    atom_pos = model.positions[idx]
    psp = model.atoms[idx].psp
    r_cut = maximum(psp.rp)
    r_cut = max(r_cut,psp.rloc)
    r_sm = r_cut*r_sm_frac
    spin = true
    charge = true
    if isnothing(target_spin)
        target_spin = 0.0
        spin = false
    end
    if isnothing(target_charge)
        target_charge = 0.0
        charge = false
    end
    @assert charge || spin
    return Constraint(atom_pos,idx,spin,charge,r_sm,r_cut,target_spin,target_charge, 0.0, 0.0, cons_resid_weight,0.0,0.0,0.0)
end

struct Constraints
    cons_vec    :: Vector{Constraint}
    overlap     :: Array{Float64,2} #overlap matrix for the different atomic functions
    overlap_inv :: Array{Float64,2}
end

function Constraints(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Constraints
    overlap = calculate_overlap(cons_vec,basis)
    overlap_inv = inv(overlap)
    unit_cell_volume = basis.model.unit_cell_volume
    for cons in cons_vec
        cons.unit_cell_volume = unit_cell_volume
    end
    return Constraints(cons_vec,overlap,overlap_inv)
end

struct ArrayAndConstraints
    """
    Struct to combine either the residual or the density with the constraints
    """
    arr         :: Array{Float64,4}
    λ           :: Array{Float64,2} #first dimension for constraints, second defines either spin or charge
    weight      :: Array{Float64,2} #define based on both cons_resid_weight and the unit cell volume
end

function ArrayAndConstraints(arr::Array{Float64,4},constraints::Constraints)::ArrayAndConstraints
    λ = zeros(Float64,length(constraints.cons_vec),2)
    weight = ones(Float64,size(λ))
    for (i,cons) in enumerate(constraints.cons_vec)
        weight[i,:] .*= cons.unit_cell_volume
    end
    return ArrayAndConstraints(arr,λ,weight)
end

Base.:+(a::ArrayAndConstraints,b::ArrayAndConstraints)= ArrayAndConstraints(a.arr + b.arr,a.λ+b.λ,a.weight)
Base.:-(a::ArrayAndConstraints,b::ArrayAndConstraints)= ArrayAndConstraints(a.arr - b.arr,a.λ-b.λ,a.weight)
Base.vec(a::ArrayAndConstraints) = vcat(vec(a.arr),vec(a.λ .* a.weight))
LinearAlgebra.norm(a::ArrayAndConstraints) = norm(a.arr) + norm(a.λ .* a.weight)

charge_density(ρ::Array{Float64,4}) = ρ[:,:,:,1]+ρ[:,:,:,2]

function weight_fn(r::AbstractVector{Float64},cons::Constraint)::Float64
    at_r = sqrt(sum((r - cons.atom_pos).^2))
    if at_r > cons.r_cut
        return 0.0
    elseif at_r < cons.r_cut-cons.r_sm
        return 1.0
    else
        x = (cons.r_cut-at_r)/cons.r_sm
        return x^2 * (3 + x*(1 + x*( -6 + 3*x)))
    end
end

function calculate_overlap(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Array{Float64,2}
    """
    integrate the overlap between the individual atomic weight functions Wᵢⱼ = ∫wᵢ(r)wⱼ(r)d
    """
    W_ij = zeros(Float64,length(cons_vec),length(cons_vec))
    overlap = zeros(Bool,length(cons_vec),length(cons_vec))
    r_vecs = collect(r_vectors(basis))

    #check if there's any overlap
    for i = 1:length(cons_vec)-1
        for j = i+1:length(cons_vec)
            r_ij = cons_vec[i].atom_pos - cons_vec[j].atom_pos
            dist = sqrt(sum(r_ij.^2))
            if dist ≤ cons_vec[i].r_cut+cons_vec[j].r_cut
                overlap[i,j] = 1
                overlap[j,i] = 1
            end
        end
    end

    #do the diagonal terms either way
    for (i,cons) in enumerate(cons_vec)
        for r in r_vecs
            W_ij[i,i] += weight_fn(r,cons)^2
        end
    end

    #if there's no overlap, then that's it. Otherwise add the off diagonal terms as necessary
    if sum(overlap) != 0
        for i = 1:length(cons_vec)
            for j = 1:length(cons_vec)
                if overlap[i,j]
                    for r in r_vecs
                        W_ij[i,j] += weight_fn(r,cons_vec[i])*weight_fn(r,cons_vec[j])
                    end
                end
            end
        end
    end
    return W_ij .* (basis.model.unit_cell_volume/prod(size(r_vecs)))
end
    
function integrate_atomic_functions(arr::Array{Float64,3},basis::PlaneWaveBasis,constraints::Constraints)::Vector{Float64}
    """
    integrate an array arr and the weight functions of each constraint
    """

    rvecs = collect(r_vectors(basis))

    spins = zeros(Float64,length(constraints.cons_vec)) #called spins since this is what you get for integrating the spin density

    for (i,cons) in enumerate(constraints.cons_vec)
        for j in eachindex(rvecs)
            w = weight_fn(rvecs[j],cons)
            if w != 0.0
                spins[i] += w * arr[j]
            end
        end
    end

    spins .*= basis.model.unit_cell_volume / prod(size(rvecs))
    return spins
end

function orthogonalise_residual!(δV::Array{Float64,3},basis::PlaneWaveBasis,constraints::Constraints,is_spin::Bool)
    """
    Orthogonalise the residual with respect to the atomic weight functions
        δV = δV -  ∑ᵢ wᵢ(r) ∑ⱼ (W)ᵢⱼ⁻¹∫δV(r')wⱼ(r')dr'
    """

    rvecs = collect(r_vectors(basis))

    δV_w_i = integrate_atomic_functions(δV,basis,constraints)

    to_be_multiplied_by_weights = constraints.overlap_inv*δV_w_i

    for (i,cons) in enumerate(constraints.cons_vec)
        for j in CartesianIndices(rvecs)
            δV[j] -=  weight_fn(rvecs[j],cons)*to_be_multiplied_by_weights[1][i]
        end
        if is_spin
            cons.λ_spin = to_be_multiplied_by_weights[i]
        else
            cons.λ_charge = to_be_multiplied_by_weights[i]
        end
    end
end      

function add_resid_constraints!(δV::Array{Float64,3},dev_from_target::Vector{Float64},constraints::Constraints,basis::PlaneWaveBasis)
    """
    The part that's added to the residual is ∑ᵢ cᵢ wᵢ(r) ∑ⱼ(W)⁻¹ᵢⱼ (Nⱼ-Nⱼᵗ)
    """
    rvecs = collect(r_vectors(basis))
    W_ij = constraints.overlap_inv
    for (i,cons) in enumerate(constraints.cons_vec)
        factor = cons.cons_resid_weight*(W_ij[i,:]⋅dev_from_target)    
        for j in CartesianIndices(rvecs)
            δV[j] += weight_fn(rvecs[j],cons)*factor
        end
    end
end

function get_spin_charge_constraints(constraints::Constraints,is_spin::Bool)::Constraints
    """
    Remove the constraints that don't constrain the specified variable (either charge or spin) defined by is_spin
    Modify the overlap matrix and the inverse overlap matrix to account as needed for the given 
    """
    
    relevant_constraints = Vector{Int}(undef,0)
    relevant_cons_vec = Vector{Constraint}(undef,0)
    for (i,cons) in enumerate(constraints.cons_vec)
        if (is_spin && cons.spin) || (!is_spin && cons.charge)
            push!(relevant_constraints,i)
            push!(relevant_cons_vec,cons)
        end
    end
    if length(relevant_cons_vec)>0
        relevant_overlap = constraints.overlap[relevant_constraints,relevant_constraints]
        relevant_overlap_inv = inv(relevant_overlap)
    else
        relevant_overlap = Array{Float64,2}(undef,0,0)
        relevant_overlap_inv = Array{Float64,2}(undef,0,0)
    end
    new_constraints = Constraints(relevant_cons_vec,relevant_overlap,relevant_overlap_inv)
    return new_constraints
end

function add_constraint_to_residual_component!(δV::Array{Float64,3},ρ::Array{Float64,3},basis::PlaneWaveBasis,constraints::Constraints,is_spin::Bool)

    tmp_constraints = get_spin_charge_constraints(constraints,is_spin)
    if length(tmp_constraints.cons_vec)>0
        current_vals = integrate_atomic_functions(ρ,basis,tmp_constraints)

        if is_spin
            current_vals .*= 2.0 #multiply by 2 to get it in Bohr Magnetons
        end

        for (i,cons) in enumerate(tmp_constraints.cons_vec)
            if is_spin
                cons.current_spin = current_vals[i]
                current_vals[i] -= cons.target_spin
            else
                cons.current_charge = current_vals[i]
                current_vals[i] -= cons.target_charge
            end
        end

        orthogonalise_residual!(δV,basis,tmp_constraints,is_spin)
        add_resid_constraints!(δV,current_vals,tmp_constraints,basis)
    end
end

function display_constraints(constraints::Vector{Constraint})

    println("Atom idx  |  Constraint Type  |  λ     |  Current Value  |  Target Value ")
    println("-------------------------------------------------------------------------")
    for cons in constraints.cons_vec
        idx = rpad(cons.atom_idx,9," ")
        if cons.spin
            λ = rpad(cons.λ_spin,6," ")[begin:6]
            current = rpad(cons.current_spin,15," ")[begin:15]
            target = rpad(cons.target_spin,13," ")[begin:13]
            println(" $idx|  spin             |  $λ|  $current|  $target")
        end
        if cons.charge
            λ = rpad(cons.λ_charge,6," ")[begin:6]
            current = rpad(cons.current_charge,15," ")[begin:15]
            target = rpad(cons.target_charge,13," ")[begin:13]
            println(" $idx|  charge           |  $λ|  $current|  $target")
        end
    end
end

display_constraints(constraints::Constraints) = display_constraints(constraints.cons_vec)

function add_constraint_to_residual!(δV::Array{Float64,4},ρ::Array{Float64,4}, basis::PlaneWaveBasis,constraints::Constraints)

    δV_charge = δV[:,:,:,1]+δV[:,:,:,2]
    δV_spin   = δV[:,:,:,1]-δV[:,:,:,2]

    ρ_charge = charge_density(ρ)
    ρ_spin = spin_density(ρ)

    add_constraint_to_residual_component!(δV_charge,ρ_charge,basis,constraints,false)
    add_constraint_to_residual_component!(δV_spin,  ρ_spin  ,basis,constraints,true )

    δV[:,:,:,1] = 0.5.*(δV_charge + δV_spin)
    δV[:,:,:,2] = 0.5.*(δV_charge - δV_spin)
end

function add_constraint_to_potential!(V::Array{Float64,4},basis::PlaneWaveBasis,constraints::Constraints)
    charge_addition = ones(Float64,size(V)[1:3])
    spin_addition   = ones(Float64,size(V)[1:3])
    charge_constraints = get_spin_charge_constraints(constraints,false)
    spin_constraints   = get_spin_charge_constraints(constraints,true)

    add_resid_constraints!(charge_addition,[cons.λ_charge for cons in charge_constraints.cons_vec],charge_constraints,basis)
    add_resid_constraints!(spin_addition,  [cons.λ_spin for cons in spin_constraints.cons_vec],    spin_constraints,basis)

    V[:,:,:,1] += charge_addition + spin_addition
    V[:,:,:,2] += charge_addition - spin_addition
end

function hamiltonian_with_constraint_and_density(ρ::ArrayAndConstraints,basis::PlaneWaveBasis,constraints::Constraints,ψ,occupation)::Hamiltonian

    ham = energy_hamiltonian(basis,ψ,occupation; ρ=ρ.arr).ham
        
    Vin = total_local_potential(ham)
    add_constraint_to_potential!(Vin,basis,constraints)
    
    ham = hamiltonian_with_total_potential(ham,Vin)

    ham
end


@timing function scf_constrained_density_mixing(
    basis::PlaneWaveBasis;
    damping=FixedDamping(0.8),
    nbandsalg::NbandsAlgorithm=AdaptiveBands(basis.model),
    fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model),
    ρ=guess_density(basis),
    V=nothing,
    ψ=nothing,
    tol=1e-6,
    maxiter=100,
    eigensolver=lobpcg_hyper,
    diag_miniter=1,
    determine_diagtol=ScfDiagtol(),
    mixing=SimpleMixing(),
    is_converged=ScfConvergenceDensity(tol),
    callback=ScfDefaultCallback(),
    acceleration=AndersonAcceleration(;m=10),
    accept_step=ScfAcceptStepAll(),
    max_backtracks=3,  # Maximal number of backtracking line searches
    constraints=nothing, # vector of Constraint structs giving constraint information
)
    """
    Hacky trial implementation of the constrained DFT density mixing approach
    Vinₙ --> ρoutₙ --> [Rₙ,{∂E/∂λₙⁱ}] --> [ρinₙ₊₁,λₙⁱ] --> Vinₙ₊₁
    The mixing is done on this combination of the density and Lagrange multipliers using the residual and the lagrange multiplier gradients
    """
    # TODO Test other mixings and lift this
    @assert (   mixing isa SimpleMixing
             || mixing isa KerkerMixing
             || mixing isa KerkerDosMixing)
    damping isa Number && (damping = FixedDamping(damping))

    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
    end

    @assert !isnothing(constraints)
    @assert typeof(constraints) == Vector{Constraint}
    constraints = Constraints(constraints,basis)

    ρ = ArrayAndConstraints(ρ,constraints)

    # Initial guess for V (if none given)
    ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ.arr).ham
    isnothing(V) && (V = total_local_potential(ham))

    function ρin2ρout(ρin; diagtol=tol / 10, ψ=nothing, eigenvalues=nothing, occupation=nothing)

        ham = hamiltonian_with_constraint_and_density(ρin,basis,constraints,ψ,occupation)

        res_V = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                             occupation, miniter=diag_miniter, tol=diagtol)
        
        (; basis, ham, ρin, energies=new_E,
         Vin, res_V...)
    end

    function residual(ρin::ArrayAndConstraints,ρout::Array{Float64,4},constraints::Constraints,basis::PlaneWaveBasis)::ArrayAndConstraints
        resid_arr = ρout - ρin.arr
        resid_λ   = zeros(Float64,size(ρin.λ))

        spin_values = integrate_atomic_functions(spin_density(ρout),basis,constraints)
        charge_values = integrate_atomic_functions(charge_density(ρout),basis,constraints)

        for (i,cons) in enumerate(constraints.cons_vec)
            if cons.charge
                resid_λ[i,1] = charge_values[i]-cons.target_charge
            end
            if cons.spin
                resid_λ[i,2] = spin_values[i] - cons.target_spin
            end
        end
        return ArrayAndConstraints(resid_arr,resid_λ,ρin.weight)
    end

    function SCF_step!(ρin,n_iter,info,constraints,basis,diagtol)
        info_next = ρin2ρout(ρin; ψ=info.guess, diagtol, info.eigenvalues, info.occupation)
        Rₙ = residual(ρin,info_next.ρout,constraints,basis)
        Rₙ.arr = mix_density(mixing, basis, Rₙ.arr; constraints, n_iter, info_next...)
        ρout =  ArrayAndConstraints(acceleration(info.ρin, info.α, Rₙ))
        update_constraints!(ρout, constraints)
        n_iter += 1
        ρin = ρout
    end

    n_iter = 1
    converged = false
    α = trial_damping(damping)
    diagtol = determine_diagtol((; ρin=ρ.arr, Vin = V, n_iter))
    info = ρin2ρout(ρ; diagtol, ψ)
    info = merge(info, (; α))

    while n_iter < maxiter
        info = merge(info, (; stage=:iterate, algorithm="SCF", converged))
        callback(info)
        if MPI.bcast(is_converged(info), 0, MPI.COMM_WORLD)
            # TODO Debug why these MPI broadcasts are needed
            converged = true
            break
        end
        SCF_step!(ρin, n_iter, info,constraints,basis,diagtol)
    end

    ham = hamiltonian_with_constraint_and_density(ρin,basis,constraints,info.guess,info.occupation)
    info = (; ham, basis, info.energies, converged, ρ=ρin.arr, info.eigenvalues,
            info.occupation, info.εF, n_iter, info.ψ, info.n_bands, info.n_bands_converge,
            info.diagonalization, stage=:finalize, algorithm="SCF",
            info.occupation_threshold)

    if !isnothing(constraints)
        constraint_info = []
        for cons in constraints.cons_vec
            λ_spin = cons.spin ? cons.λ_spin : nothing
            λ_charge = cons.charge ? cons.λ_charge : nothing
            
            target_spin = cons.spin ? cons.target_spin : nothing
            target_charge = cons.charge ? cons.target_charge : nothing

            current_spin = cons.spin ? cons.current_spin : nothing
            current_charge = cons.charge ? cons.current_charge : nothing

            atom_idx = cons.atom_idx
            push!(constraint_info, (; atom_idx, target_spin, target_charge, λ_spin, λ_charge, current_spin, current_charge))
        end
        info = merge(info, (; constraint_info))
    end
    callback(info)
    info
end
