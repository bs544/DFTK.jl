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
    # current_spin      :: Float64
    # current_charge    :: Float64
    cons_resid_weight :: Float64
    # λ_charge          :: Float64 #Lagrange multiplier for constraint
    # λ_spin            :: Float64
end

function Constraint(model::Model,idx::Int,cons_resid_weight::Float64=1.0,r_sm_frac::Float64=0.05;r_cut=nothing,target_spin=nothing,target_charge=nothing)::Constraint
    atom_pos = vector_red_to_cart(model,model.positions[idx])
    psp = model.atoms[idx].psp
    if isnothing(r_cut)
        r_cut = maximum(psp.rp)
        r_cut = max(r_cut,psp.rloc)
    end
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
    return Constraint(atom_pos,idx,spin,charge,r_sm,r_cut,target_spin,target_charge, cons_resid_weight)
end

struct Constraints
    # make all the constraint information readily accessible, also precompute whatever you can to save on time

    cons_vec      :: Vector{Constraint} #kept if any specific information is needed
    overlap_charge:: Array{Float64,2}   #overlap matrix for the different atomic functions
    overlap_spin  :: Array{Float64,2}
    at_fn_arrays  :: Array{Array{Float64,3},2} # precomputed atomic functions
    res_wgt_arrs  :: Array{Float64,2}          # weights assigned to the lagrange multiplier updates
    lambdas       :: Array{Float64,2}          # lagrange multipliers
    is_constrained:: Array{Int64,2}            # mask for whether a constraint is applied
    target_values :: Array{Float64,2}          # target values, 0 if unconstrained
    current_values:: Array{Float64,2}          # current values, 0 if unconstrained
    dvol          :: Float64                   # just basis.dvol, useful when integrating
end

function Constraints(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Constraints
    atomic_fns = get_at_function_arrays(cons_vec,basis)
    overlap_charge,overlap_spin = calculate_overlap(atomic_fns,basis.dvol)

    res_wgt_arrs  = zeros(Float64,(length(cons_vec),2))
    lambdas       = zeros(Float64,(length(cons_vec),2))
    is_constrained= zeros(Float64,(length(cons_vec),2))
    target_values = zeros(Float64,(length(cons_vec),2))
    current_values= zeros(Float64,(length(cons_vec),2))

    for (i,cons) in enumerate(cons_vec)
        if cons.charge
            res_wgt_arrs[i,1]   = cons.resid_weight
            is_constrained[i,1] = 1
            target_values[i,1]  = cons.target_charge
        end
        if cons.spin
            res_wgt_arrs[i,2]   = cons.resid_weight
            is_constrained[i,2] = 1
            target_values[i,2]  = cons.target_spin
        end

    return Constraints(cons_vec,overlap_charge,overlap_spin,atomic_fns,res_wgt_arrs,lambdas,is_constrained,target_values,current_values,basis.dvol)
end

function periodic_dist(r::AbstractVector{Float64},basis::PlaneWaveBasis)::Float64
    #presumably a way better way of doing this somehow
    red_vector = vector_cart_to_red(basis.model,r)
    #the closest image will be in one of the 27 unit cells, so just focus on them
    disp = [1.0,0.0,-1.0]
    cell_disp = Array{Vector{Float64},3}(undef,3,3,3)
    for i=1:3; for j = 1:3; for k=1:3
        cell_disp[i,j,k] = [disp[i],disp[j],disp[k]]
    end;end;end
    red_vector_options = map(x->x+red_vector,cell_disp)
    red_vector_options = vector_red_to_cart.(basis.model,red_vector_options)
    red_vector_dists = norm.(red_vector_options)
    return minimum(red_vector_dists)
end

function weight_fn(r::AbstractVector{Float64},cons::Constraint,basis::PlaneWaveBasis)::Float64
    at_r = periodic_dist(r-cons.atom_pos,basis)
    if at_r > cons.r_cut
        return 0.0
    elseif at_r < cons.r_cut-cons.r_sm
        return 1.0
    else
        x = (cons.r_cut-at_r)/cons.r_sm
        return x^2 * (3 + x*(1 + x*( -6 + 3*x)))
    end
end

function get_at_function_arrays(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Array{Array{Float64,3},2}
    """
    Calculate the weight functions just the once in an effort to speed things up.
    """
    r_vecs = collect(r_vectors_cart(basis))

    asize = size(r_vecs)
    cons_asize = (length(cons_vec),2)
    weights = [zeros(Float64,asize) for i=1:cons_asize[1], j=1:cons_asize[2]]
    for (i,cons) in enumerate(cons_vec)
        w_arr = zeros(Float64,asize)
        for j in eachindex(r_vecs)
            w_arr[j] = weight_fn(r_vecs[j],cons,basis)
        end
        if cons.charge
            weights[i,1] = w_arr
        end
        if cons.spin
            weights[i,2] = w_arr
        end
    end
    return weights
end

function calculate_overlap(atom_fns::Array{Array{Float64,3},2},dvol::Float64)::Tuple{Array{Float64,2},Array{Float64,2}}
    """
    If the atomic functions have been committed to arrays, then the integration is a lot cheaper
    """
    n_constraint = size(atom_fns)[1]
    overlap_spin   = zeros(Float64,(n_constraint,n_constraint))
    overlap_charge = zeros(Float64,(n_constraint,n_constraint))
    for i = 1:n_constraint
        for j = i:n_constraint
            overlap_spin[i,j]   = sum(atom_fns[i,2].*atom_fns[j,2])
            overlap_charge[i,j] = sum(atom_fns[i,1].*atom_fns[j,1])

            overlap_spin[j,i]   = overlap_spin[i,j]
            overlap_charge[j,i] = overlap_charge[i,j]
        end
    end
    overlap_spin   .*= dvol
    overlap_charge .*= dvol

    return overlap_charge,overlap_spin

end

function integrate_atomic_functions(arr::Array{Float64,3},constraints::Constraints,idx::Int)::Vector{Float64}
    """
    Use the atomic functions to integrate an array arr
    """
    arr_values = Vector{Float64}(undef,length(constraints.cons_vec))
    for i = 1:length(constraints.cons_vec)
        arr_values[i] = sum(constraints.at_fn_arrays[i,idx].*arr)*constraints.dvol
    end
    return arr_values
end

function inv_with_zeros(arr::Array{Float64,2})::Array{Float64,2}
    """
    When taking the inverse of the overlap matrix, some of the columns and rows will be zero,
    remove these, then take the inverse and add them back in
    """
    nonzero_columns = [i for i = 1:size(arr)[1] if sum(arr[i,:])!=0]
    
    nonzero_arr = arr[nonzero_columns,nonzero_columns]
    inv_nonzero = inv(nonzero_arr)
    inv_arr = zeros(Float64,size(arr))
    for (i,idx) in enumerate(nonzero_columns)
        for (j,jdx) in enumerate(nonzero_columns)
            inv_arr[idx,jdx] = inv_nonzero[i,j]
        end
    end
    return inv_arr
end

function orthogonalise_residual!(δV::Array{Float64,3},constraints::Constraints,is_spin::Bool)
    """
    Orthogonalise the residual with respect to the atomic weight functions
        δV = δV -  ∑ᵢ wᵢ(r) ∑ⱼ (W)ᵢⱼ⁻¹∫δV(r')wⱼ(r')dr'
    """
    if is_spin
        spin_idx = 2
        overlap_inv = inv_with_zeros(constraints.overlap_spin)
    else
        spin_idx = 1
        overlap_inv = inv_with_zeros(constraints.overlap_charge)
    end
    residual_atomic_components = integrate_atomic_functions(δV,constraints.at_fn_arrays,spin_idx)
    to_be_multiplied_by_weights = overlap_inv*residual_atomic_components

    for i = 1:length(constraints.cons_vec)
        δV -= constraints.at_fn_arrays[i,spin_idx].*to_be_multiplied_by_weights[i]
        constraints.lambdas[i,spin_idx] = to_be_multiplied_by_weights[i]
    end
end

function add_resid_constraints!(δV::Array{Float64,3},dev_from_target::Vector{Float64},constraints::Constraints,spin_idx::Int)
    """
    The part that's added to the residual is ∑ᵢ cᵢ wᵢ(r) ∑ⱼ(W)⁻¹ᵢⱼ (Nⱼ-Nⱼᵗ)
    """
    overlap_inv = spin_idx==1 ? inv_with_zeros(constraints.overlap_charge) : inv_with_zeros(constraints.overlap_spin)
    for (i,cons) in enumerate(constraints.cons_vec)
        factor = constraints.weights[i,spin_idx]*(overlap_inv[i,:]⋅dev_from_target)
        δV += constraints.at_fn_arrays[i,spin_idx].*factor
    end
end

function add_constraint_to_residual_component!(δV::Array{Float64,3},ρ::Array{Float64,3},constraints::Constraints,is_spin::Bool)
    """
    For either the spin or the charge, get the current atomic values, then orthogonalise the residual wrt. the atomic functions and add the difference between the target and current values
    """
    spin_idx = is_spin ? 2 : 1
    if sum(constraints.is_constrained[:,spin_idx]) > 0
        current_vals = integrate_atomic_functions(ρ,constraints,spin_idx)

        constraints.current_values[:,spin_idx] = current_vals

        deriv_array = (current_vals - constraints.target_values[:,spin_idx]).*constraints.is_constrained[:,spin_idx]

        orthogonalise_residual!(δV,constraints,is_spin)
        add_resid_constraints!(δV,deriv_array,constraints,spin_idx)
    end
end

function display_constraints(constraints::Constraints)

    println("Atom idx  |  Constraint Type  |  λ     |  Current Value  |  Target Value ")
    println("-------------------------------------------------------------------------")
    for (i,cons) in enumerate(constraints.cons_vec)
        idx = rpad(cons.atom_idx,9," ")
        if cons.spin
            λ = rpad(constraints.lambdas[i,2],6," ")[begin:6]
            current = rpad(constraints.current_values[i,2],15," ")[begin:15]
            target = rpad(constraints.target_values[i,2],13," ")[begin:13]
            println(" $idx|  spin             |  $λ|  $current|  $target")
        end
        if cons.charge
            λ = rpad(constraints.lambdas[i,1],6," ")[begin:6]
            current = rpad(constraints.current_values[i,1],15," ")[begin:15]
            target = rpad(constraints.target_values[i,1],13," ")[begin:13]
            println(" $idx|  charge           |  $λ|  $current|  $target")
        end
    end
end

function add_constraint_to_residual!(δV::Array{Float64,4},ρ::Array{Float64,4},constraints::Constraints)

    δV_charge = δV[:,:,:,1]+δV[:,:,:,2]
    δV_spin   = δV[:,:,:,1]-δV[:,:,:,2]

    ρ_charge = total_density(ρ)
    ρ_spin = spin_density(ρ)

    add_constraint_to_residual_component!(δV_charge,ρ_charge,constraints,false)
    add_constraint_to_residual_component!(δV_spin,  ρ_spin  ,constraints,true )

    δV[:,:,:,1] = 0.5.*(δV_charge + δV_spin)
    δV[:,:,:,2] = 0.5.*(δV_charge - δV_spin)

    # display_constraints(constraints.cons_vec)
end


