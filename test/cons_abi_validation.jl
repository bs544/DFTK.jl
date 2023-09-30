using DFTK
include("testcases.jl")

function run_iron_abi_cons_validation()

    test_spins    = [6.0000000000E-01,
                    8.0000000000E-01,
                    1.0000000000E+00,
                    1.2000000000E+00,
                    1.4000000000E+00,
                    1.6000000000E+00,
                    1.8000000000E+00,
                    2.0000000000E+00,
                    2.2000000000E+00,
                    2.4000000000E+00]

    test_energies = [-1.1864478190E+02,
                    -1.1864396831E+02, 
                    -1.1864277574E+02, 
                    -1.1864106414E+02, 
                    -1.1863863037E+02, 
                    -1.1863516550E+02, 
                    -1.1863018943E+02, 
                    -1.1862313216E+02, 
                    -1.1861376159E+02, 
                    -1.1860222810E+02]

    test_grspins  = [3.2905756354E-03,
                    4.9204540120E-03,
                    7.1213711326E-03,
                    1.0160053517E-02,
                    1.4429494922E-02,
                    2.0614838199E-02,
                    2.9681647799E-02,
                    4.1101813285E-02,
                    5.2378335915E-02,
                    6.3043817716E-02]

    kgrid = [11,11,11]
    Ecut  = 20  #Hartree
    fft_size = [20,20,20]
    n_bands = 14
    r_sm  = 0.05 #Bohr
    r_cut = 2.00 #Bohr
    r_sm_frac = r_sm/r_cut
    resid_weight = 1.0
    damping_param=0.8

    grspins = Dict{Int,Float64}()
    energies = Dict{Int,Float64}()
    # psp = PspHgh("./test/testcases_ABINIT/cons_iron_PBE/Fe-q16-pbe.abinit.hgh";identifier="Fe-q16-pbe.abinit")
    
    magnetic_moments = [3.0]
    Fe        = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/pbe/Fe-q16.hgh"))
    model     = model_PBE(iron_bcc.lattice, [Fe], iron_bcc.positions;
                          temperature=0.01, magnetic_moments) 
    basis     = PlaneWaveBasis(model;Ecut,kgrid,fft_size)
    nbandsalg = AdaptiveBands(basis.model, n_bands_converge=n_bands)

    V=nothing


    for i in 1:length(test_spins)
        spin = test_spins[i]
        ρ0 = guess_density(basis,[spin])
        println("Spin: $spin")
        constraints = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
        scfres = DFTK.scf_potential_mixing(basis;tol=1.0e-10,ρ=ρ0,V=V,constraints=constraints,maxiter=100,damping=DFTK.FixedDamping(damping_param),nbandsalg);
        energies[i]=scfres.energies.total
        grspins[i]=scfres.constraints.lambdas[1,2]/2

        println(scfres.constraints.current_values[1,2])
    end
    println("spin \t abi_energy \t dftk_energy \t abi_grspin \t dftk_grspin \t energy_diff \t grspin_diff")
    println("--------------------------------------------------------------------------------------------")
    for i in 1:length(test_spins)
        spin = test_spins[i]
        abi_energy = test_energies[i]
        abi_grspin = test_grspins[i]
        dftk_energy= energies[i]
        dftk_grspin= grspins[i]
        ediff = abi_energy-dftk_energy
        grspin_diff = abi_grspin-dftk_grspin
        println("$spin:\t $abi_energy \t $dftk_energy \t $abi_grspin \t $dftk_grspin \t $ediff \t $grspin_diff")

    end
    
    
end

run_iron_abi_cons_validation()