"""
functions needed to constrain some target properties. e.g. atomic spin

The potential mixing constrained DFT method only requires a change to the residual before preconditioning

The density mixing constrained DFT method is a bit more involved.

NOTE: Charge constraints are harder to converge. Try smaller mixing parameters and smaller weights for the constrained part of the residual
"""

    


function orthogonalise_residual!(δV::Array{Float64,3},basis::PlaneWaveBasis,constraints::Constraints,is_spin::Bool)
    """
    Orthogonalise the residual with respect to the atomic weight functions
        δV = δV -  ∑ᵢ wᵢ(r) ∑ⱼ (W)ᵢⱼ⁻¹∫δV(r')wⱼ(r')dr'
    """

    rvecs = collect(r_vectors_cart(basis))

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
    rvecs = collect(r_vectors_cart(basis))
    W_ij = constraints.overlap_inv
    for (i,cons) in enumerate(constraints.cons_vec)
        factor = cons.cons_resid_weight*(W_ij[i,:]⋅dev_from_target)    
        for j in CartesianIndices(rvecs)
            δV[j] += weight_fn(rvecs[j],cons)*factor
        end
    end
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

function add_constraint_to_residual!(δV::Array{Float64,4},ρ::Array{Float64,4}, basis::PlaneWaveBasis,constraints::Constraints)

    δV_charge = δV[:,:,:,1]+δV[:,:,:,2]
    δV_spin   = δV[:,:,:,1]-δV[:,:,:,2]

    ρ_charge = total_density(ρ)
    ρ_spin = spin_density(ρ)

    add_constraint_to_residual_component!(δV_charge,ρ_charge,basis,constraints,false)
    add_constraint_to_residual_component!(δV_spin,  ρ_spin  ,basis,constraints,true )

    δV[:,:,:,1] = 0.5.*(δV_charge + δV_spin)
    δV[:,:,:,2] = 0.5.*(δV_charge - δV_spin)
end


