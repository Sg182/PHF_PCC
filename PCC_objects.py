"""
This file contains code for the PCC settings and Values objects
"""

import numpy as np
import json
import warnings


class settings:
    """
    Object carrying settings handed to program
    """

    def __init__(self, file, args):
        self.get_settings(file, args)
        self.validate_settings()

    def get_settings(self, file, args):
        """
        Read in the optimization settings
        Settings priorities are as follows:
            1) Command line argument(when applicable)
            2) json file handed to program
            3) settings.json file(if it exists in the current directory)
            4) Defaults hard coded below

        PHF Settings
        -----------------------------------------
        mol_name : str
            name of molecule, used for checkpoint files
        geom_file : str
            xyz file containing molecular geometry(Not hard coded, I'm not guessing the molecule)
        basis : str or list
            basis set to use(Not hard coded, I'm not guessing the basis)
            If a list of basis sets are given, PHF will be projected from one to the next
        unit : str
            Unit for geometry optimization(use something pyscf can understand)
        orb_init : str
            Name of binary file containing the AO orbitals to use as the
            initial guess of PHF. They should be in the basis listed first in
            settings.basis and represented in the spin orbital basis, alpha
            first then beta.
        VERBOSE : str
            Controls the amount of printing to the stdout
        HamType : str
            Type of Hamiltonian, 'Mol' for molecular Hamiltonian and 'Hub' for Hubbard
        chkpoint : bool
            Create pickled objects for checkpointing
        read_PHF : bool
            Read pickled object containing PHF orbitals
            For scan calculations, affects first step only
        read_PCC : bool
            Read pickled object containing PCC amplitudes
            For scan calculations, affects first step only
        refine_PHF : bool
            When reading PHF results, run through optimizer to improve PHF
            For scan calculations, affects first step only
        use_MPI : bool
            Use MPI parallelization
        is_RHF : bool
            Use if starting orbitals are RHF
        fixgauge : bool
            Adjust deform determinant gauge to improve PCC optimization
        SP : int
            Determine type of spin projection
                SP = 0, No projection
                SP = 1, SGHF
                SP = 2, SUHF
                SP = 3, SzPHF
        ngrid : list
            Number of gridpoint for Lebedev quadrature, list of two integers.
            For SUHF or SzPHF, grid should be set as [1,n] for any integer n.
        J : int
            Desired J value from spin projection
        M : int
            Desired M value from spin projection
        PG : str
            Point group to project into, 'None' for no projection
        Irrep : str
            Irrep of point group to project into
        CmplxConj : int
            Complex Conj = 0, real
                         = 1, complex
                         = 2, complex projection
        NFC : int
            Number of frozen core orbitals, freezes lowest energy orbitals
        NFV : int
            Number of frozen virtual orbitals, removes highest energy orbitals
        NFO : int
            ? Leave as 0
        scan : int
            Scan PES along direction specified in xyz file

        Hubbard Settings
        -----------------------------------------
        U0 : float
            Interaction strength in Hamiltonian
        nx : int
            Number of sites in x direction
        ny : int
            Number of sites in y direction
        nele : int
            Number of electrons on lattice
        kx : int
            Number of ?
        ky : int
            Number of ?

        Convergence Parameters for Optimization
        -----------------------------------------
        PHF_maxiter : int
            Max number of iterations in SCF or DIIS optimization of PHF orbitals
        PHF_thrsh : float
            Convergence toloerance in SCF(energy) or DIIS(error) optimization of PHF orbitals
        PHF_opt : str
            Local optimizer for the direct (Thouless-parameter) PHF minimization in optPHF.
            Production policy (validated 2026-09-28 on N2 and square H4, cc-pVDZ):
                "auto" (default): SP = 2 (SUHF)  -> IPOPT, PHF_basinhop hops (default 1)
                                  SP = 1 (SGHF)  -> BFGS + basin hopping with max(PHF_basinhop, 3) hops
                                  other SP       -> BFGS + PHF_basinhop hops
                "ipopt"         : IPOPT (cyipopt, L-BFGS Hessian, bounds +/-5 on z) as the local
                                  optimizer for any SP -- deterministic when PHF_basinhop = 0;
                                  for SGHF it is a validation/local option, not the default global
                                  search (for the tested H4 seeds it stayed in a higher basin unless
                                  >= 3 hops were used)
                "bfgs"          : scipy BFGS (gtol = PHF_bfgs_gtol) + PHF_basinhop hops
            Every optimizer result is validated independently (recomputed E and G): finite
            parameters/energy/gradient, max|G| <= 10 * (PHF_ipopt_tol | PHF_bfgs_gtol), energy not
            above the starting energy, |z| below the bound.  The optimizer's own success flag is
            never trusted by itself; a failed validation raises RuntimeError.
        PHF_bfgs_gtol : float
            BFGS gradient tolerance (scipy gtol, max|dE/dz|), default 1e-5
        PHF_sghf_kick : float
            SP = 1 only: if the reference determinant is collinear (every occupied orbital pure
            alpha or beta), a deterministic perturbation of this size is applied to the spin-flip
            block of the initial Thouless parameters before optimization.  Reason: a collinear
            determinant is an Sz eigenstate and therefore a stationary point of the projected
            energy in all spin-flip directions, and the finite-difference behaviour of the
            gradient exactly there (with CmplxConj = 1) is not validated.  Production SGHF must
            always start off the collinear manifold; 0 disables the kick (not recommended).
        PHF_ipopt_tol : float
            IPOPT convergence tolerance (unscaled dual infeasibility = max|dE/dz|) for IPOPT
        PHF_basinhop : int
            Number of scipy basin-hopping restarts wrapped around the local minimizer selected by
            PHF_opt (random displacement of the Thouless parameters, then re-minimize, Metropolis
            accept).  0 = a single local minimization from the seed (deterministic).  Default 1
            (the legacy behaviour for BFGS).  Works with both "bfgs" and "ipopt".
        PHF_hop_step : float
            Basin-hopping displacement size (default 0.2)
        PHF_hop_T : float
            Basin-hopping Metropolis temperature in Hartree (default 0.2)

        PCC Settings
        ------------------------------------------
        do_CC : bool
            Do PCC after PHF
        nBroyVec : int
            Number of vectors to store in Broyden minimization
        """

        type_map = {
            bool: "boolean",
            int: "integer",
            float: "float",
            str: "string",
            list: "list",
            complex: "complex",
            object: "object",
            tuple: "tuple",
        }

        "list of settings to look for and their respective type"
        sets = [
            ("do_CC", bool),
            ("VERBOSE", int),
            ("geom_file", str),
            ("basis", str, list),
            ("unit", str),
            ("mol_name", str),
            ("orb_init", str),
            ("HamType", str),
            ("U0", float),
            ("nx", int),
            ("ny", int),
            ("nele", int),
            ("NFC", int),
            ("NFV", int),
            ("NFO", int),
            ("is_RHF", bool),
            ("chkpoint", bool),
            ("read_PHF", bool),
            ("read_PCC", bool),
            ("refine_PHF", bool),
            ("use_MPI", bool),
            ("fixgauge", bool),
            ("SP", int),
            ("ngrid", list),
            ("M", int),
            ("J", int),
            ("PG", str),
            ("Irrep", str),
            ("kx", int),
            ("ky", int),
            ("CmplxConj", int),
            ("nBroyVec", int),
            ("PHF_maxiter", int),
            ("PHF_thrsh", float),
            ("PHF_opt", str),
            ("PHF_ipopt_tol", float),
            ("PHF_bfgs_gtol", float),
            ("PHF_sghf_kick", float),
            ("PHF_basinhop", int),
            ("PHF_hop_step", float),
            ("PHF_hop_T", float),
            ("DIIS", bool),
            ("scan", int),
        ]

        keys = [x[0] for x in sets]

        "Set defaults"
        setattr(self, "do_CC", False)
        setattr(self, "VERBOSE", 1)
        setattr(self, "unit", "ang")
        setattr(self, "orb_init", None)
        setattr(self, "HamType", "Mol")
        setattr(self, "NFC", 0),
        setattr(self, "NFV", 0),
        setattr(self, "NFO", 0),
        setattr(self, "is_RHF", False),
        setattr(self, "chkpoint", False),
        setattr(self, "read_PHF", False),
        setattr(self, "read_PCC", False),
        setattr(self, "refine_PHF", False),
        setattr(self, "use_MPI", False),
        setattr(self, "fixgauge", True),
        setattr(self, "SP", 0),
        setattr(self, "PG", "None"),
        setattr(self, "Irrep", "Ag"),
        setattr(self, "nx", 0),
        setattr(self, "ny", 0),
        setattr(self, "kx", 1),
        setattr(self, "ky", 1),
        setattr(self, "CmplxConj", 0),
        setattr(self, "nBroyVec", 30),
        setattr(self, "PHF_maxiter", 100),
        setattr(self, "PHF_thrsh", 1.e-4),
        setattr(self, "PHF_opt", "auto"),
        setattr(self, "PHF_ipopt_tol", 1.e-6),
        setattr(self, "PHF_bfgs_gtol", 1.e-5),
        setattr(self, "PHF_sghf_kick", 1.e-2),
        setattr(self, "PHF_basinhop", 1),
        setattr(self, "PHF_hop_step", 0.2),
        setattr(self, "PHF_hop_T", 0.2),
        setattr(self, "DIIS", True),
        setattr(self, "scan", 0),

        "Read settings.json"
        if not file == "settings.json":
            try:
                with open("settings.json", "r") as f:
                    data = f.read()
                    temp = json.loads(data)

                for prop, *typ in sets:
                    if f"{prop}" in temp.keys():
                        if not type(temp[f"{prop}"]) in typ:
                            raise ValueError(
                                f"{prop} must be a {type_map[typ]}, error in settings.json"
                            )
                        setattr(self, f"{prop}", temp[f"{prop}"])

                "Check for extra options"
                for key in temp.keys():
                    if not key in keys:
                        raise Exception(f"Received extra option {key} from settings.json")

            except FileNotFoundError:
                print("No settings.json file found.")

        "read custom json file"
        if not file == None:
            try:
                with open(file, "r") as f:
                    data = f.read()
                    temp = json.loads(data)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Custom settings json file does not exist! If no custom settings desired,"
                    "then do not ask for them, dummy!"
                )

            for prop, *typ in sets:
                if f"{prop}" in temp.keys():
                    if not type(temp[f"{prop}"]) in typ:
                        raise ValueError(
                            f"{prop} must be a {type_map[typ]}, error in {file}"
                        )
                    setattr(self, f"{prop}", temp[f"{prop}"])

            "Check for extra options"
            for key in temp.keys():
                if not key in keys:
                    raise Exception(f"Received extra option {key} from {file}")

        "read in command line arguments"
        if type(args) == object:
            if args.geometry:
                setattr(self, "geom_file", args.geometry)
            if args.basis:
                setattr(self, "basis", args.basis)

        return

    def validate_settings(self):
        """Check that the recieved settings are valid"""

        "Check for required settings"
        if not "geom_file" in self.__dict__.keys():
            raise Exception("Must specify a xyz file with geometry")
        if not "basis" in self.__dict__.keys():
            raise Exception("Must specify a basis for calculation")
        "If basis was given as string, change it to a list"
        if type(self.basis) == str:
            setattr(self, "basis", [self.basis])

        if not self.HamType in ["Mol", "Hub"]:
            raise Exception("HamType must be either 'Mol' or 'Hub'")

        "Hubbard Checks"
        if self.HamType == "Hub":
            if not "U0" in self.__dict__.keys():
                raise Exception("Must specify U0 for Hubbard calculations")
            if not "nx" in self.__dict__.keys():
                raise Exception("Must specify nx for Hubbard calculations")
            if not "ny" in self.__dict__.keys():
                raise Exception("Must specify ny for Hubbard calculations")
            if not "nele" in self.__dict__.keys():
                raise Exception("Must specify nele for Hubbard calculations")
            if not "kx" in self.__dict__.keys():
                raise Exception("Must specify kx for Hubbard calculations")
            if not "ky" in self.__dict__.keys():
                raise Exception("Must specify ky for Hubbard calculations")
            if self.nele < 0:
                raise Exception("Number of electrons must be non-negative")
            if self.nx < 0:
                raise Exception("nx must be at least 1")
            if self.ny < 0:
                raise Exception("ny must be at least 1")
            if self.kx < 1:
                raise Exception("kx must be at least 1")
            if self.ky < 1:
                raise Exception("ky must be at least 1")

        "Convergence checks"
        if (self.PHF_thrsh < 0):
            raise ValueError("PHF_thrsh must be a positive number")
        if (self.PHF_maxiter < 1):
            raise ValueError("PHF_maxiter must be at least 1")
        if self.PHF_opt not in ("auto", "ipopt", "bfgs"):
            raise ValueError('PHF_opt must be "auto", "ipopt" or "bfgs"')
        if (self.PHF_ipopt_tol <= 0) or (self.PHF_bfgs_gtol <= 0):
            raise ValueError("PHF_ipopt_tol and PHF_bfgs_gtol must be positive numbers")
        if (self.PHF_sghf_kick < 0):
            raise ValueError("PHF_sghf_kick must be >= 0")
        if (self.PHF_basinhop < 0):
            raise ValueError("PHF_basinhop must be >= 0")
        if (self.PHF_hop_step <= 0) or (self.PHF_hop_T <= 0):
            raise ValueError("PHF_hop_step and PHF_hop_T must be positive")

        "Projection checks"
        if (self.SP < 0) or (self.SP > 3):
            raise ValueError("SP must be between 0 and 3")
        if (self.SP == 2 or self.SP == 3) and (self.ngrid[0] != 1):
            raise Exception("Only SGHF uses both gridpoints")
        if (self.SP == 0 and ((self.ngrid[0] != 1) or (self.ngrid[1] != 1))):
            raise Exception("ngrid should be [1,1] when not doing spin projection")
        if (self.SP > 0) and (
            (not "J" in self.__dict__.keys()) or (not "M" in self.__dict__.keys())
        ):
            raise Exception("Must specify desired J and M values for spin projection")

        if (not self.PG == "None") and (not "Irrep" in self.__dict__.keys()):
            raise Exception("Must specify desired Irrep for point group projection")

        if (self.SP > 0) and (self.J < 0):
            raise ValueError("J values must be non-negative")
        if (self.SP > 0) and (abs(self.M) > self.J):
            raise ValueError("|J| must be less than or equal to M")
        if (self.CmplxConj < 0) or (self.CmplxConj > 2):
            raise ValueError("SP must be between 0 and 2")

        if self.nBroyVec < 1:
            raise ValueError("nBroyVec must be greater than 0")

        if self.NFC < 0:
            raise ValueError("NFC must not be negative")
        if self.NFV < 0:
            raise ValueError("NFV must not be negative")
        if self.NFO < 0:
            raise ValueError("NFO must not be negative")

        if (self.chkpoint == True) and (not "mol_name" in self.__dict__.keys()):
            raise Exception("Must specify a mol_name to use for checkpointing")

        if self.VERBOSE:
            print("Control Settings:")
            for opt in self.__dict__:
                print(f"{opt} = {getattr(self, opt)}")

        return


class Values(object):
    """
    Object carrying often used quantities to clean up name space
    """

    def __init__(self):
        """
        cMF Values
        -----------------------------------------
        mol : pyscf mol object
            pyscf mol object for molecule
        Enuc : float
            Nuclear repulsion energy
        h1 : float array
            Core hamiltonian
        eri : float array
            electron repulsion integrals
        NAO : float array
            Number of AO
        NOccSO : float array
            Total number of electrons
        """
        setattr(self, "mol", (object))
        setattr(self, "Enuc", (float))
        setattr(self, "h1", (float))
        setattr(self, "eri", (float))
        setattr(self, "NAO", (int))
        setattr(self, "NOccSO", (int))
        setattr(self, "ncisp", (int))
        setattr(self, "roota", (int))
        setattr(self, "rootb", (int))
        setattr(self, "rooty", (int))
        setattr(self, "weightsp", (int))
        setattr(self, "R1", (int))
        setattr(self, "R2", (int))
        setattr(self, "SGHFMO", (int))
        setattr(self, "fsp", (int))
        setattr(self, "fpg", (int))
        setattr(self, "fk", (int))
        setattr(self, "EPHF", (int))
        setattr(self, "OAO", (int))
