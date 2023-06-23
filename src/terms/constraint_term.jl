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


# ene_ops(args...;ρ::ArrayAndConstraints,kwargs...) = ene_ops(args...;ρ.arr,kwargs...)


struct DensityMixingConstraints
    cons_vec :: Vector{Constraint}
end

struct TermDensityConstraint 
    constraints::Constraints
end

function (constraints::DensityMixingConstraints)(basis::PlaneWaveBasis)
    return TermDensityConstraint(Constraints(constraints.cons_vec,basis))
end



@timing "ene_ops: constraint" function ene_ops(term::TermDensityConstraint,basis::PlaneWaveBasis,ψ,occupation;ρ::Array,cons_λ::Array,cons_weight::Array)
    ρ = ArrayAndConstraints(ρ,cons_λ,cons_weight)

    #update the constraints
    for (i,cons) in enumerate(term.constraints.cons_vec)
        cons.λ_charge = ρ.λ[i,1]
        cons.λ_spin = ρ.λ[i,2]
    end
    #define the constrained potential array
    pot_size = basis.fft_size
    if basis.model.spin_polarization==:collinear
        pot_size = (pot_size...,2)
    end
    constrained_potential=zeros(Float64,pot_size)
    

    charge_constraint_pot = zeros(Float64,basis.fft_size)
    charge_constraints = get_spin_charge_constraints(term.constraints,false)
    add_constraint_to_arr!(charge_constraint_pot,basis,charge_constraints,false)


    if length(pot_size)==4 # then spin and charge can be constrained, so do the spin too
        
        spin_constraint_pot = zeros(Float64,basis.fft_size)
        spin_constraints   = get_spin_charge_constraints(term.constraints,true)
        add_constraint_to_arr!(spin_constraint_pot,basis,spin_constraints,true)

        constrained_potential[:,:,:,1] = 0.5.*(charge_constraint_pot+spin_constraint_pot)
        constrained_potential[:,:,:,2] = 0.5.*(charge_constraint_pot-spin_constraint_pot)
    else
        constrained_potential = charge_constraint_pot
    end

    ops = [RealSpaceMultiplication(basis, kpt, constrained_potential[:,:,:,kpt.spin])
           for kpt in basis.kpoints]
    E = sum(total_density(ρ.arr) .* constrained_potential) * basis.dvol

    (; E, ops)
end




