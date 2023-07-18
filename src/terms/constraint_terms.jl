"""
Adds the constrained potential to the hamiltonian given the fields in the ArrayAndConstraints struct used for the density with the constraints.
    I tried overloading the ene_ops function so that it would pass the array only for the rest of the ene_ops functions, but that didn't work, so I'm explicitly naming the constraint parts now

The way these things seem to work is that you define a struct with parameters that define the potential. This is the struct that's fed into the Model struct
After that you have a second struct that basically does the ready made potential evaluation. 
    This struct is made by calling the first struct as a function with the basis as the argument, so you have to define that function too

Something like
struct DensityMixingConstraints
    constraints :: Constraints
end

struct TermConstraint
    constraints:: Constraints
end

function (constraints::DensityMixingConstraints)(basis::PlaneWaveBasis)
    return TermConstraint(constraints.constraints)
end

function ene_ops(term::TermConstraint,args...,kwargs...)
    return pot
end

Also need to include the force computation. Stress computation is done by forward differentiation

"""

struct DensityMixingConstraint
    """
    wrapper for the vector of constraints. This is what's given as the model constrainint term
    """
    cons_vec :: Vector{Constraint}
end

struct TermDensityMixingConstraint
    constraints::Constraints
end

function (dm_constraints::DensityMixingConstraint)(basis::PlaneWaveBasis)
    return TermDensityMixingConstraint(Constraints(dm_constraints.cons_vec,basis))
end

@timing "ene_ops: constraint" function ene_ops(term::TermDensityMixingConstraint,basis,ψ,occupation;ρ::Array,cons_lambdas,kwargs...)

    #update the constraints
    term.constraints.lambdas = cons_lambdas

    current_charges = integrate_atomic_functions(total_density(ρ),term.constraints,1)
    current_spins   = integrate_atomic_functions(spin_density(ρ) ,term.constraints,2)

    term.constraints.current_values[:,1] = current_charges
    term.constraints.current_values[:,2] = current_spins

    charge_pot_mod = zeros(Number,size(term.constraints.at_fn_arrays[1,1]))
    spin_pot_mod   = zeros(Number,size(term.constraints.at_fn_arrays[1,1]))

    for i=1:length(term.constraints.cons_vec)
        charge_pot_mod += term.constraints.at_fn_arrays[i,1].*term.constraints.lambdas[i,1].*term.constraints.is_constrained[i,1]
        spin_pot_mod   += term.constraints.at_fn_arrays[i,2].*term.constraints.lambdas[i,2].*term.constraints.is_constrained[i,2]
    end

    # if basis.model.spin_polarization==:collinear
    #     pot_size = (size(charge_pot_mod)...,2)
    # else
    #     pot_size = size(charge_pot_mod)
    # end
    pot_size = size(ρ)

    pot_mod = zeros(Number,pot_size)

    if length(pot_size)==4 && pot_size[4] == 2
        pot_mod[:,:,:,1] = (charge_pot_mod+spin_pot_mod)#.*0.5
        pot_mod[:,:,:,2] = (charge_pot_mod-spin_pot_mod)#.*0.5
    elseif length(pot_size)==4
        pot_mod[:,:,:,1] = charge_pot_mod
    else
        pot_mod = charge_pot_mod
    end

    ops = [RealSpaceMultiplication(basis,kpt,pot_mod[:,:,:,kpt.spin])
           for kpt in basis.kpoints]
    
    E = sum(term.constraints.lambdas .* (term.constraints.current_values.-term.constraints.target_values))

    (; E, ops)

end

apply_kernel(term::TermDensityMixingConstraint,args...;kwargs...) = nothing
