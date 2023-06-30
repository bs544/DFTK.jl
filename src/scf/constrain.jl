"""
Functions needed to change the potential residual before preconditioning. 
The Constraint and Constraints structs are defined in ../constraint_types.jl
"""


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

function orthogonalise_residual(δV::Array{Float64,3},constraints::Constraints,is_spin::Bool)
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
    residual_atomic_components = integrate_atomic_functions(δV,constraints,spin_idx)
    to_be_multiplied_by_weights = overlap_inv*residual_atomic_components
    for i = 1:length(constraints.cons_vec)
        δV -= constraints.at_fn_arrays[i,spin_idx].*to_be_multiplied_by_weights[i]
        constraints.lambdas[i,spin_idx] = to_be_multiplied_by_weights[i]
    end
    return δV
end

function add_resid_constraints(δV::Array{Float64,3},dev_from_target::Vector{Float64},constraints::Constraints,spin_idx::Int)
    """
    The part that's added to the residual is ∑ᵢ cᵢ wᵢ(r) ∑ⱼ(W)⁻¹ᵢⱼ (Nⱼ-Nⱼᵗ)
    """
    overlap_inv = spin_idx==1 ? inv_with_zeros(constraints.overlap_charge) : inv_with_zeros(constraints.overlap_spin)
    for (i,cons) in enumerate(constraints.cons_vec)
        factor = constraints.res_wgt_arrs[i,spin_idx]*(overlap_inv[i,:]⋅dev_from_target)
        δV += constraints.at_fn_arrays[i,spin_idx].*factor
    end
    return δV
end

function add_constraint_to_residual_component(δV::Array{Float64,3},ρ::Array{Float64,3},constraints::Constraints,is_spin::Bool)
    """
    For either the spin or the charge, get the current atomic values, then orthogonalise the residual wrt. the atomic functions and add the difference between the target and current values
    """
    spin_idx = is_spin ? 2 : 1
    if sum(constraints.is_constrained[:,spin_idx]) > 0
        current_vals = integrate_atomic_functions(ρ,constraints,spin_idx)

        constraints.current_values[:,spin_idx] = current_vals

        deriv_array = (current_vals - constraints.target_values[:,spin_idx]).*constraints.is_constrained[:,spin_idx]
        δV = orthogonalise_residual(δV,constraints,is_spin)
        δV = add_resid_constraints(δV,deriv_array,constraints,spin_idx)
    end
    return δV
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

function add_constraint_to_residual(δV::Array{Float64,4},ρ::Array{Float64,4},constraints::Constraints)

    δV_charge = δV[:,:,:,1]+δV[:,:,:,2]
    δV_spin   = δV[:,:,:,1]-δV[:,:,:,2]

    ρ_charge = total_density(ρ)
    ρ_spin = spin_density(ρ)

    δV_charge = add_constraint_to_residual_component(δV_charge,ρ_charge,constraints,false)
    δV_spin   = add_constraint_to_residual_component(δV_spin,  ρ_spin  ,constraints,true )

    δV[:,:,:,1] = 0.5.*(δV_charge + δV_spin)
    δV[:,:,:,2] = 0.5.*(δV_charge - δV_spin)

    # display_constraints(constraints) #for debugging purposes
    return δV
end


