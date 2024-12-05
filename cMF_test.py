#!/usr/bin/python

"""
This code performs the cMF optimization for molecules using the algebraic approach, run through the driver function below.

To use this code, prepare molecular information by either making a gaussian .mat file with the geometry or
the geometry can be input directly into pyscf. The molecule choice is controlled by the parameter 'molecule'.
A list of molecules available are listed below, add more as needed.

For examples of using the .mat file, follow the code for instances of molecule > 0.
For an example of providing the geometry directly, look at the H2 cluster example of molecule = -2.
Hubbard is also available as molecule = -1

Algorithmic details are controlled with parameters in driver()

Print level is controlled with VERBOSE, default is 1. 0 prints only energies. 2 prints some smaller things. 3 prints bigger things. 4 prints very big things. 5 A genuinely unnecessary amount of printing.
"""
Molecules = {
    -20: "F()",
    -20: "Neon Dimer",
    -15: "Ar(+8) ion",
    -14: "Cl(+7) ion",
    -13: "S(+6) ion",
    -12: "P(+5) ion",
    -11: "Si(+4) ion",
    -10: "Al(+3) ion",
    -9: "Mg(+2) ion",
    -8: "Na(+1) ion",
    -7: "Neon atom",
    -6: "F(-1) ion",
    -5: "H2 cluster(polyhedron)",
    -4: "H2 cluster(polygon)",
    -3: "H2 cluster(linear)",
    -2: "H2 cluster(stack)",
    -1: "Hubbard",
    1: "butadiene sto-3g",
    2: "butadiene cc-pvdz",
    3: "benzene sto-3g",
    4: "benzene cc-pvdz",
    5: "hexatriene sto-3g",
    6: "hexatriene cc-pvdz",
    7: "C12 cc-pvdz",
    8: "C16 cc-pvdz",
    9: "C20 cc-pvdz",
    10: "Coronene cc-pvdz",
    11: "Coronene alt cc-pvdz(lower energy)",
    12: "Circumcoronene cc-pvdz",
    13: "tetraene cc-pvdz",
    14: "pentaene cc-pvdz",
    15: "pyrene cc-pvdz",
    16: "pyrene alt cc-pvdz(lower energy)",
    17: "C54 cc-pvdz",
    18: "C44 cc-pvdz",
    19: "C30 cc-pvdz",
    20: "C24 cc-pvdz",
    21: "ethyl",
    22: "tetraene sto-3g",
}

import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as ssl
import scipy.special as ss
import h5py as h5

from pyscf import gto, scf, lo, mcscf, fci, ao2mo

# import QCMatEl as qcm
# import QCOpMat as qco

np.set_printoptions(precision=14, threshold=np.inf, suppress=True, linewidth=200000000)
VERBOSE = 1


def get_mol_param(molecule):

    """Get molecule information for prepared systems, all systems are assumed to have
       the pi orbitals in the z direction and the conjugated chain in the x-y plane.

    param -- carries the mat file where the geometry is stored, the basis set, N, and M
    orb_swap -- used when the active space is determined by a CAS calculation, it is a list of
                pairs of orbitals to be swaped from the RHF calculation to put the active
                orbitals nearest to the HOMO. Put and empty list if not needed
    atom_list -- is a list of the atoms in the conjugated chain with bonded atoms adjacent in the list

    See documentation for available instances"""

    # fmt: off
    param = [["diene_6g.mat","sto-3g",4,4],["diene_2z.mat","cc-pvdz",4,4],
             ["benzene_3g.mat","sto-3g",6,6],["benzene_2z.mat","cc-pvdz",6,6],
             ["triene_6g.mat","sto-3g",6,6],["triene_6g.mat","cc-pvdz",6,6],
             ["C12.mat","cc-pvdz",12,12],["C16.mat","cc-pvdz",16,16],["C20.mat","cc-pvdz",20,20],
             ["coro.mat","cc-pvdz",24,24],["coro.mat","cc-pvdz",24,24],
             ["circoro.mat","cc-pvdz",54,54],["tetraene.mat","cc-pvdz",8,8],
             ["pentaene.mat","cc-pvdz",10,10],["pyrene_geom.mat","cc-pvdz",16,16],["pyrene_geom.mat","cc-pvdz",16,16],
             ["C54.mat","cc-pvdz",54,54],["C44.mat","cc-pvdz",44,44],["C30.mat","cc-pvdz",30,30],["C24.mat","cc-pvdz",24,24],["ethyl.mat","cc-pvdz",2,2],
             ["tetraene_1z.mat","sto-3g",8,8]]
    orb_swap = [[],[[16,19]],[[16,18]],[[16,18],[23,29]],[],[[23,29]],[[45,53],[47,56],[48,57]],[[60,70],[61,73],[62,75],[64,76]],[],[],[],[],[[31,36],[32,39]],[[38,42],[39,46],[40,47]],[],[],[],[],[],[],[],[]]
    atom_list = [[0,3,5,7],[0,3,5,7],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,3,5,7,9,11],[0,3,5,7,9,11],
            [0,3,5,7,9,11,13,15,17,19,21,23],[0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39],
            [0,1,2,3,4,5,23,6,12,13,7,14,15,8,16,17,9,18,19,10,20,21,11,22],
            [3,9,10,4,18,19,5,11,22,23,6,0,1,7,14,15,8,2,17,16,13,12,21,20],
            [0,1,2,3,4,5,12,6,23,48,51,53,7,13,27,25,24,14,15,8,16,31,59,30,9,17,34,61,35,18,19,10,20,41,39,36,11,21,44,63,47,22,56,54,58,28,32,60,37,62,42,45,49,64],
            [0,3,5,7,9,11,13,15],[0,3,5,7,9,11,13,15,17,19],
            [0,1,6,10,11,7,12,13,8,14,15,9,4,5,2,3],[2,7,11,10,6,1,8,3,4,9,15,14,12,13,0,5],#,[2,7,12,13,8,3,11,10,6,1,0,5,4,9,15,14],#[1,2,7,11,10,6,3,4,9,15,14,8,5,0,13,12],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47],[0,3],[0,3,5,7,9,11,13,15]]
    # fmt: on

    "Important parameters"
    mel_file, basis, N, M = param[molecule - 1]

    "Get info from matfile to build mol"
    mel = qcm.MatEl(file=mel_file)
    coord = mel.c.reshape((-1, 3), order="C")
    atm_chg = mel.atmchg
    nocc = mel.ne // 2
    no = M
    ic0 = nocc - N // 2
    mol = gto.Mole()
    mol.atom = []
    for i in range(len(atm_chg)):
        mol.atom.append([int(atm_chg[i]), tuple(coord[i, :])])
    #    mol.unit = "angstrom"
    mol.unit = "au"
    mol.basis = basis
    mol.max_memory = 110000

    mol.build()
    
    atoms = atom_list[molecule - 1]
    swap = orb_swap[molecule - 1]

    if VERBOSE > 1:
        print("Molecular Parameters:")
        print(f"Molecule specification from {mel_file}")
        print(f"Doing a CAS({N},{M}) calculation")
        print(f"Using the {basis} basis")
        print("Molecule Coordinates")
        print(coord)
        print(f"{ic0} core orbitals")
        print("Conjugated atoms:")
        print(atoms)

    "Use CAS for active space?"
    if not molecule in [7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20]:
        CAS = True
    else:
        CAS = False

    return mol, N, M, atoms, swap, CAS, ic0, nocc


def find_atoms(mol, M, atoms):

    """Locate atoms and orbitals in pi system"""

    """Make list of atoms in chain and pz orbitals"""
    xbst = mol._bas[:, 1]
    nctr = mol._bas[:, 3]
    xbs = np.empty(0)
    atm = np.empty(0)
    for i in range(len(nctr)):
        for _ in range(nctr[i]):
            xbs = np.hstack((xbs, xbst[i]))
            atm = np.hstack((atm, mol._bas[i, 0]))
    xbs = np.array([int(x) for x in xbs])
    atm = np.array([int(x) for x in atm])

    if np.allclose(np.sort(atoms), np.arange(M)):
        carbons = atoms
    else:
        carbons = np.ones(M, dtype=int) * -1
        ind = -1
        used = []
        for i in range(M):
            val = M * 100
            for j in range(M):
                if (atoms[j] < val) and (not j in used):
                    ind = j
                    val = atoms[j]
            carbons[ind] = i
            used.append(ind)

    "Find 2pz orbitals"
    pz = []
    used = []
    x = 0
    for i in range(len(xbs)):
        if (xbs[i] == 1) and (atm[i] in atoms) and (not atm[i] in used):
            pz.append(x + 2)
            used.append(atm[i])
        x += 2 * xbs[i] + 1
    if len(pz) != M:
        raise

    if VERBOSE > 2:
        print("List of carbon atoms:")
        print(carbons)
        print("List of pz orbitals:")
        print(pz)

    return carbons, pz

def auto_cluster(mol, val_N=0):

    """make an initial guess for clusters"""
    """val_N :: the valence n quantum number"""

    xbst = mol._bas[:, 1]
    nctr = mol._bas[:, 3]
    xbs = np.empty(0)
    atm = np.empty(0)
    for i in range(len(nctr)):
        for _ in range(nctr[i]):
            xbs = np.hstack((xbs, xbst[i]))
            atm = np.hstack((atm, mol._bas[i, 0]))
    xbs = np.array([int(x) for x in xbs])
    atm = np.array([int(x) for x in atm])

    "Find 2s,2p---3s,3p orbitals"
    act_orb=[]
    x=0
    
    for i in range(len(xbs)):
        if xbs[i]==0 and 0<x<3:
            act_orb.append(x)
        elif xbs[i]==1:
            act_orb.extend([x,x+3,x+1,x+4,x+2,x+5])
            break    
        
        x+= 2*xbs[i]+1

    return act_orb


def prepare_cf(molecule, unrestrict, n=None, FCI=False):

    """Prepare active orbitals"""
    if molecule > 0:
        if VERBOSE > 1:
            print("Getting molecular parameters")
        mol, N, M, atoms, swap, CAS, ic0, nocc = get_mol_param(molecule)

        S = mol.intor_symmetric("int1e_ovlp")
        nvir = mol.nao - nocc
        if VERBOSE > 2:
            print("AO Overlap:")
            print(S)

        if VERBOSE > 1:
            print("Locating pz orbitals")
        carbons, pz = find_atoms(mol, M, atoms)

        if CAS:
            if VERBOSE:
                print("Using CAS for active space")
            "Get orbitals from CAS calculation"
            mf = scf.RHF(mol).run()
            C = mf.mo_coeff
            for x in swap:
                C[:, x] = C[:, list(reversed(x))]

            "Print to check orbital choice"
            if VERBOSE > 1:
                print(f"{ic0} core orbitals")
                print(C[:, :ic0])
                print("Active space for CAS")
                print(C[pz, ic0 : ic0 + N])

            cas = mcscf.CASSCF(mf, N, M).run()
            C = cas.mo_coeff
            if VERBOSE:
                print("CAS Energy", cas.e_cas)

            "Core orbitals for later"
            cf0 = C.copy()

            "Make ideal orbitals"
            D = np.zeros((mol.nao, M))
            for i in range(M // 2):
                D[pz[carbons[2 * i]], 2 * i] = 1
                D[pz[carbons[2 * i + 1]], 2 * i] = 1
                D[pz[carbons[2 * i]], 2 * i + 1] = 1
                D[pz[carbons[2 * i + 1]], 2 * i + 1] = -1
            S_ = D.T.dot(S).dot(D)
            val, vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
            D = D.dot(S_)
            if VERBOSE > 2:
                print("Ideal Orbitals:")
                print(D)

            "Project ideal orbitals onto active space"
            Cac = C[:, ic0 : ic0 + M]
            Dac = Cac.dot(Cac.T).dot(S).dot(D)
            S_ = Dac.T.dot(S).dot(Dac)
            val, vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
            cf = Dac.dot(S_)
            if VERBOSE > 1:
                print("Active Orbitals to be used")
                print(cf[pz, :])

            U = np.eye(M)

        else:
            "Get orbitals from UHF orbitals"
            if VERBOSE:
                print("Using UHF for active space")

            "Get UHF guess with broken symmetry"
            mf = scf.RHF(mol).run()
            C = mf.mo_coeff
            T = np.zeros((M), dtype=int)
            q = 0
            di = np.diag(C[pz, :].T.dot(S[pz, :][:, pz]).dot(C[pz, :]))
            for i in range(mol.nao):
                if di[i] > 0.1:
                    T[q] = i
                    q += 1
                    if q == M:
                        break
            if len(T) != M:
                raise
            cf0 = np.zeros_like(C)
            q = 0
            for i in range(nocc):
                if not i in T:
                    cf0[:, q] = C[:, i]
                    q += 1

            Cup = np.zeros((mol.nao, M))
            Cdn = np.zeros((mol.nao, M))
            for i in range(M // 2):
                Cup[pz[carbons[2 * i]], i] = 1
                Cdn[pz[carbons[2 * i + 1]], i] = 1
            for i in range(M // 2):
                Cup[pz[carbons[2 * i + 1]], i + M // 2] = 1
                Cdn[pz[carbons[2 * i]], i + M // 2] = 1

            S_ = Cup.T.dot(S).dot(Cup)
            val, vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
            Cup = Cup.dot(S_)

            S_ = Cdn.T.dot(S).dot(Cdn)
            val, vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
            Cdn = Cdn.dot(S_)

            "Orthonomalize up and down guess"
            D = C.copy()
            for i in range(M):
                C[:, T[i]] = Cup[:, i]
                D[:, T[i]] = Cdn[:, i]

            S_ = C.T.dot(S).dot(C)
            val, vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
            C = C.dot(S_)

            S_ = D.T.dot(S).dot(D)
            val, vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
            D = D.dot(S_)

            mocc = np.zeros((mol.nao))
            mocc[:nocc] = 1

            "Run UHF"
            mfu = scf.UHF(mol).newton()
            dm = mfu.make_rdm1((C, D), (mocc, mocc))
            mfu = mfu.run(dm)
            Cup, Cdn = mfu.mo_coeff

            "Build charge density and diagonalize for orbitals"
            chg_den = Cup[:, :nocc].dot(Cup[:, :nocc].T) + Cdn[:, :nocc].dot(
                Cdn[:, :nocc].T
            )
            chg_den = chg_den.dot(S)
            val, vec = sl.eig(chg_den)
            val = np.real(val)
            if VERBOSE > 1:
                print("Charge Density Eigenvalues:")
                print(val)
            vec = np.real(vec)
            T = []
            V = []
            U = []
            for i in range(len(val)):
                if (val[i] > 0.01) and (val[i] < 1.99):
                    T.append(i)
                elif val[i] > 1.99:
                    V.append(i)
            T = np.array(T)
            if len(T) != M:
                raise Exception("Wrong number of Active orbitals found")
            if len(V) != ic0:
                raise Exception("Wrong number of core orbitals found")
            Cac = vec[:, T]
            cf0 = vec[:, V]
            S_ = np.diag(1.0 / np.sqrt(np.diag(cf0.T.dot(S).dot(cf0))))

            "Core Orbitals"
            cf0 = cf0.dot(S_)

            if not unrestrict:
                "Restricted guess"

                "Make ideal orbitals"
                D = np.zeros((mol.nao, M))
                for i in range(M // 2):
                    D[pz[carbons[2 * i]], 2 * i] = 1
                    D[pz[carbons[2 * i + 1]], 2 * i] = 1
                    D[pz[carbons[2 * i]], 2 * i + 1] = 1
                    D[pz[carbons[2 * i + 1]], 2 * i + 1] = -1
                S_ = D.T.dot(S).dot(D)
                val, vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
                D = D.dot(S_)
                if VERBOSE > 2:
                    print("Ideal Orbitals:")
                    print(D)

                "Project ideal orbitals onto active space"
                Dac = Cac.dot(Cac.T).dot(S).dot(D)
                S_ = Dac.T.dot(S).dot(Dac)
                val, vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
                cf = Dac.dot(S_)
                if VERBOSE > 1:
                    print("Active orbitals to be used")
                    print(cf[pz, :])

                U = np.eye(M)

            else:
                "Unrestricted guess"

                "Find up actives in occupied"
                T = np.zeros((M // 2), dtype=int)
                q = 0
                di = np.diag(
                    Cup[:, :nocc][pz, :]
                    .T.dot(S[pz, :][:, pz])
                    .dot(Cup[:, :nocc][pz, :])
                )
                for i in range(nocc):
                    if di[i] > 0.1:
                        T[q] = i
                        q += 1
                        if q == M // 2:
                            break
                if q != M // 2:
                    raise

                "Find up actives in virtual"
                V = np.zeros((M // 2), dtype=int)
                q = 0
                di = np.diag(
                    Cup[:, nocc:][pz, :]
                    .T.dot(S[pz, :][:, pz])
                    .dot(Cup[:, nocc:][pz, :])
                )
                for i in range(mol.nao - nocc):
                    if di[i] > 0.1:
                        V[q] = i
                        q += 1
                        if q == M // 2:
                            break
                if q != M // 2:
                    raise
                V += nocc

                Cup = np.hstack((Cup[:, T], Cup[:, V]))

                "Find down actives in occupied"
                T = np.zeros((M // 2), dtype=int)
                q = 0
                di = np.diag(
                    Cdn[:, :nocc][pz, :]
                    .T.dot(S[pz, :][:, pz])
                    .dot(Cdn[:, :nocc][pz, :])
                )
                for i in range(nocc):
                    if di[i] > 0.1:
                        T[q] = i
                        q += 1
                        if q == M // 2:
                            break
                if q != M // 2:
                    raise

                "Find down actives in virtual"
                V = np.zeros((M // 2), dtype=int)
                q = 0
                di = np.diag(
                    Cdn[:, nocc:][pz, :]
                    .T.dot(S[pz, :][:, pz])
                    .dot(Cdn[:, nocc:][pz, :])
                )
                for i in range(mol.nao - nocc):
                    if di[i] > 0.1:
                        V[q] = i
                        q += 1
                        if q == M // 2:
                            break
                if q != M // 2:
                    raise
                V += nocc

                Cdn = np.hstack((Cdn[:, T], Cdn[:, V]))

                "Make ideal orbitals"
                Dup = np.zeros((mol.nao, M))
                Ddn = np.zeros((mol.nao, M))
                for i in range(M // 2):
                    Dup[pz[carbons[2 * i]], i] = 1
                    Ddn[pz[carbons[2 * i + 1]], i] = 1
                for i in range(M // 2):
                    Dup[pz[carbons[2 * i + 1]], M // 2 + i] = 1
                    Ddn[pz[carbons[2 * i]], M // 2 + i] = 1

                "Project ideal up orbitals onto guess"
                Dup = Cup.dot(Cup.T).dot(S).dot(Dup)
                S_ = Dup.T.dot(S).dot(Dup)
                val, vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
                Dup = Dup.dot(S_)

                "Project ideal down orbitals onto guess"
                Ddn = Cdn.dot(Cdn.T).dot(S).dot(Ddn)
                S_ = Ddn.T.dot(S).dot(Ddn)
                val, vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
                Ddn = Ddn.dot(S_)

                "Project ideal orbitals onto active space"
                Dac = Cac.dot(Cac.T).dot(S).dot(Dup)
                S_ = Dac.T.dot(S).dot(Dac)
                val, vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
                cf = Dac.dot(S_)

                Dac = Cac.dot(Cac.T).dot(S).dot(Ddn)
                S_ = Dac.T.dot(S).dot(Dac)
                val, vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1.0 / np.sqrt(val))).dot(vec.T)
                cfdn = Dac.dot(S_)

                if VERBOSE > 1:
                    print("Cluster orbitals")
                    print("up")
                    print(cf[pz, :])
                    print("down")
                    print(cfdn[pz, :])

                "U maps from cf to cfdn"
                U = cfdn.T.dot(S).dot(cf)
                U = U.T
                if VERBOSE > 1:
                    print("Map from up to down")
                    print(U)

    elif molecule == -1:
        "Hubbard model"

        "len of chain is m"
        n = 7
        m = 4 * n + 2
        if VERBOSE:
            print("Length of chain is ", m)

        U = 2
        T = 1
        Hc = np.zeros((m, m))
        Hc[:-1, 1:] -= np.eye(m - 1) * T
        Hc += Hc.T
        Hc[0, -1] = Hc[-1, 0] = -T
        eri = np.zeros((m, m, m, m))
        for i in range(m):
            eri[i, i, i, i] = U
        cf = np.eye(m)
        nb = m
        enr = 0.0
        no = m
        U = np.eye(m)
        cf0 = None
        ic0 = None
        mf = None
        mol=None # Shishir made the changes here
        N = M = m
        S = np.eye(m)
        if VERBOSE > 1:
            print("Starting orbitals:")
            print(cf)
        if VERBOSE > 3:
            print("Core Hamiltonian:")
            print(Hc)
            print("Two Particle Hamiltonian:")
            print(eri)

    elif molecule < -1 and molecule > -6:
        "H2 cluster(-2:stack,-3:linear,-4:polygon,-5:polyhedron)"
        "number of H atoms"
        if n == None:
            n = 4

        "Distance between groups(size of inner polygon for polygon and polyhedron)"
        dis = 3.0
        "bond length of each H2"
        bond = 1.4

        if n > 12 and (molecule != -4):
            raise Exception("Code needs to be update for more than 6 molecules")
        mol = gto.Mole()
        mol.atom = []
        "Define molecule"
        if molecule == -2:
            if VERBOSE:
                print(f"H2 cluster(stacked) with {n//2} molecules")
            mol.atom.append([1, (-bond / 2, -dis, 0.0)])
            mol.atom.append([1, (+bond / 2, -dis, 0.0)])
            mol.atom.append([1, (-bond / 2, 0.0, 0.0)])
            mol.atom.append([1, (+bond / 2, 0.0, 0.0)])
            if n > 4:
                mol.atom.append([1, (-bond / 2, +dis, 0.0)])
                mol.atom.append([1, (+bond / 2, +dis, 0.0)])
            if n > 6:
                mol.atom.append([1, (-bond / 2, +dis * 2, 0.0)])
                mol.atom.append([1, (+bond / 2, +dis * 2, 0.0)])
            if n > 8:
                mol.atom.append([1, (-bond / 2, +dis * 3, 0.0)])
                mol.atom.append([1, (+bond / 2, +dis * 3, 0.0)])
            if n > 10:
                mol.atom.append([1, (-bond / 2, +dis * 4, 0.0)])
                mol.atom.append([1, (+bond / 2, +dis * 4, 0.0)])
        if molecule == -3:
            if VERBOSE:
                print(f"H2 cluster(linear) with {n//2} molecules")
            dis += bond
            mol.atom.append([1, (-bond / 2, -0.0, 0.0)])
            mol.atom.append([1, (+bond / 2, -0.0, 0.0)])
            mol.atom.append([1, (-bond / 2 + dis, 0.0, 0.0)])
            mol.atom.append([1, (+bond / 2 + dis, 0.0, 0.0)])
            if n > 4:
                mol.atom.append([1, (-bond / 2 + 2 * dis, +0.0, 0.0)])
                mol.atom.append([1, (+bond / 2 + 2 * dis, +0.0, 0.0)])
            if n > 6:
                mol.atom.append([1, (-bond / 2 + 3 * dis, +0.0, 0.0)])
                mol.atom.append([1, (+bond / 2 + 3 * dis, +0.0, 0.0)])
            if n > 8:
                mol.atom.append([1, (-bond / 2 + 4 * dis, +0.0, 0.0)])
                mol.atom.append([1, (+bond / 2 + 4 * dis, +0.0, 0.0)])
            if n > 10:
                mol.atom.append([1, (-bond / 2 + 5 * dis, +0.0, 0.0)])
                mol.atom.append([1, (+bond / 2 + 5 * dis, +0.0, 0.0)])
        if molecule == -4:
            if VERBOSE:
                print(f"H2 cluster(polygon) with {n//2} molecules")
            m = n // 2
            dis /= 2.0

            "Make inner/outer polygons"
            ang = [a * 2 * np.pi / m for a in range(m)]
            inner = np.zeros((2 * m, 3))
            for i in range(m):
                inner[i, :] = np.array([np.cos(ang[i]), np.sin(ang[i]), 0.0])
            outer = (dis + bond) * inner
            inner *= dis
            for i in range(m):
                mol.atom.append([1, tuple(inner[i, :])])
                mol.atom.append([1, tuple(outer[i, :])])
        if molecule == -5:
            if VERBOSE:
                print(f"H2 cluster(polyhedron) with {n//2} molecules")
            if n == 10:
                raise Exception(
                    "There is no regular polyhedron with five vertices dummy!!"
                )
            if n < 8:
                raise Exception(
                    "A regular polyhedron with fewer than 4 vertices is a polygon, I will not repeat myself!!"
                )
            m = n // 2
            dis /= 2.0

            if m == 4:
                "Make tetrahedron"
                inner = np.array(
                    [
                        [+1.0, +1.0, +1.0],
                        [+1.0, -1.0, -1.0],
                        [-1.0, +1.0, -1.0],
                        [-1.0, -1.0, +1.0],
                    ]
                ) / np.sqrt(3)
            elif m == 6:
                "Make octahedron"
                inner = np.array(
                    [
                        [+1.0, +0.0, +0.0],
                        [-1.0, +0.0, +0.0],
                        [+0.0, +1.0, +0.0],
                        [+0.0, -1.0, +0.0],
                        [+0.0, +0.0, +1.0],
                        [+0.0, +0.0, -1.0],
                    ]
                )

            outer = (dis + bond) * inner
            inner *= dis
            for i in range(m):
                mol.atom.append([1, tuple(inner[i, :])])
                mol.atom.append([1, tuple(outer[i, :])])

        mol.unit = "ang"
        #        mol.unit = 'au'
        mol.basis = "sto-6g"
        if VERBOSE:
            print(f"Using {mol.basis}")
        mol.build()
        if FCI:
            if VERBOSE:
                print(f"Computing FCI for comparison")
            mf = scf.RHF(mol).run()
            ci = fci.FCI(mf)
            print("FCI Energy: ", ci.kernel()[0])

        if VERBOSE > 1:
            print("Molecular Coordinates:")
            print(mol.atom)

        cf0 = np.zeros((n, n))
        nb, no = cf0.shape
        cf0[0, 0] = +1.0 / np.sqrt(2.0)
        cf0[1, 0] = +1.0 / np.sqrt(2.0)
        cf0[0, 1] = +1.0 / np.sqrt(2.0)
        cf0[1, 1] = -1.0 / np.sqrt(2.0)

        if n > 2:
            cf0[2, 2] = +1.0 / np.sqrt(2.0)
            cf0[3, 2] = +1.0 / np.sqrt(2.0)
            cf0[2, 3] = +1.0 / np.sqrt(2.0)
            cf0[3, 3] = -1.0 / np.sqrt(2.0)
        if n > 4:
            cf0[4, 4] = +1.0 / np.sqrt(2.0)
            cf0[5, 4] = +1.0 / np.sqrt(2.0)
            cf0[4, 5] = +1.0 / np.sqrt(2.0)
            cf0[5, 5] = -1.0 / np.sqrt(2.0)
        if n > 6:
            cf0[6, 6] = +1.0 / np.sqrt(2.0)
            cf0[7, 6] = +1.0 / np.sqrt(2.0)
            cf0[6, 7] = +1.0 / np.sqrt(2.0)
            cf0[7, 7] = -1.0 / np.sqrt(2.0)
        if n > 8:
            cf0[8, 8] = +1.0 / np.sqrt(2.0)
            cf0[9, 8] = +1.0 / np.sqrt(2.0)
            cf0[8, 9] = +1.0 / np.sqrt(2.0)
            cf0[9, 9] = -1.0 / np.sqrt(2.0)
        if n > 10:
            cf0[10, 10] = +1.0 / np.sqrt(2.0)
            cf0[11, 10] = +1.0 / np.sqrt(2.0)
            cf0[10, 11] = +1.0 / np.sqrt(2.0)
            cf0[11, 11] = -1.0 / np.sqrt(2.0)

        if VERBOSE > 2:
            print("Ideal Orbitals:")
            print(cf0)
        S = mol.intor_symmetric("int1e_ovlp")
        M = np.dot(cf0.T, np.dot(S, cf0))
        
        val, vec = sl.eigh(-M)
        U = np.dot(vec * 1.0 / np.sqrt(-val), vec.T)

        cf = np.dot(cf0, U)
        N, M = n, n
        U = np.eye(M)
        cf0 = None
        ic0 = None
        mf = None
        if VERBOSE > 1:
            print("Initial Orbitals:")
            print(cf)

    elif molecule < -5 and molecule > -20:


        """
        Neon-like atoms
        """
        mol=gto.M()
        mol.atom=[]


        # if molecule == -6:
        #     mol.atom.append([10,(0.0,0.0,0.0)])

        #     if True:
        #          mol.basis='cc-pvdz'
        #         # mol.basis={'Ne': gto.basis.parse('''
        #         # Ne    S
        #         #     1.788000E+04           7.380000E-04          -1.720000E-04           0.000000E+00
        #         #     2.683000E+03           5.677000E-03          -1.357000E-03           0.000000E+00
        #         #     6.115000E+02           2.888300E-02          -6.737000E-03           0.000000E+00
        #         #     1.735000E+02           1.085400E-01          -2.766300E-02           0.000000E+00
        #         #     5.664000E+01           2.909070E-01          -7.620800E-02           0.000000E+00
        #         #     2.042000E+01           4.483240E-01          -1.752270E-01           0.000000E+00
        #         #     7.810000E+00           2.580260E-01          -1.070380E-01           0.000000E+00
        #         #     1.653000E+00           1.506300E-02           5.670500E-01           0.000000E+00
        #         #     4.869000E-01          -2.100000E-03           5.652160E-01           1.000000E+00
        #         # Ne    P
        #         #     2.839000E+01           4.608700E-02           0.000000E+00
        #         #     6.270000E+00           2.401810E-01           0.000000E+00
        #         #     1.695000E+00           5.087440E-01           0.000000E+00
        #         #     4.317000E-01           4.556600E-01           1.000000E+00
        #         # ''')}
        #     else:
                
                # mol.basis={'Ne': gto.basis.parse('''
                # Ne    S
                #     1.788000E+04           7.380000E-04          -1.720000E-04           0.000000E+00
                #     2.683000E+03           5.677000E-03          -1.357000E-03           0.000000E+00
                #     6.115000E+02           2.888300E-02          -6.737000E-03           0.000000E+00
                #     1.735000E+02           1.085400E-01          -2.766300E-02           0.000000E+00
                #     5.664000E+01           2.909070E-01          -7.620800E-02           0.000000E+00
                #     2.042000E+01           4.483240E-01          -1.752270E-01           0.000000E+00
                #     7.810000E+00           2.580260E-01          -1.070380E-01           0.000000E+00
                #     1.653000E+00           1.506300E-02           5.670500E-01           0.000000E+00
                #     4.869000E-01          -2.100000E-03           5.652160E-01           1.000000E+00
                # Ne    P
                #     2.839000E+01           4.608700E-02           0.000000E+00
                #     6.270000E+00           2.401810E-01           0.000000E+00
                #     1.695000E+00           5.087440E-01           0.000000E+00
                #     4.317000E-01           4.556600E-01           1.000000E+00
                # Ne    D
                #     2.202000E+00           1.0000000
                # ''')}

        if molecule < -5:
            n=-7-molecule
            mol.atom.append([10+n,(0.0,0.0,0.0)])
            # print(n)
            mol.charge =+n 
            mol.basis='cc-pvtz'


        if VERBOSE:
            print(f"Using {mol.basis}")
        
        mol.unit = 'ang'
        # mol.unit = 'au'
        mol.build()
        # print(f"Charge in the atom {mol.charge}") 
        #TODO corrections for trial
        mf = scf.RHF(mol).run()
        orbital_matrix=mf.mo_coeff
        # print("Orbital Matrix: ",orbital_matrix[:,:4])
                      
        if FCI:
            if VERBOSE:
                print(f"Computing FCI for comparison")
            mf = scf.RHF(mol).run()
            ci = fci.FCI(mf)
            print("FCI Energy: ", ci.kernel()[0])
            

        if VERBOSE > 1:
            print("Molecular Coordinates:")
            print(mol.atom)

        n = 5 #number of cluster * 2: number of orbital/2
        
        nb = mol.nao
        no = 9 #1s,2s,3-2p,3s,3-3p,5-3d
        
        cf0=np.zeros((nb,no)) # in case of just atom use nb,norb
        nb, no = cf0.shape

        act_orb=auto_cluster(mol)

        cf0[:,0]=orbital_matrix[:,0]
        for i,j in enumerate(act_orb):
            cf0[j,i+1]=1
   
        # print(cf0[:,:3])
        # print(cf0[:,3:])
        
        
        # cf1=np.zeros_like(cf0)
        # cf1[:,0]=orbital_matrix[:,0]        
        # cf1[1,1]=1
        # cf1[2,2]=1
        # cf1[3,3]=1
        # cf1[6,4]=1
        # cf1[4,5]=1
        # cf1[7,6]=1
        # cf1[5,7]=1
        # cf1[8,8]=1
        
        # print(cf1[:,:3])
        # print(cf1[:,3:])
        # print(cf0==cf1)
        


        if VERBOSE > 2:
            print("Ideal Orbitals:")
            print(cf0)

        S = mol.intor_symmetric("int1e_ovlp")
        
        #projection
        P=np.eye(nb)-cf0[:,0:1]@cf0[:,0:1].T@S
        P_cf0=P@cf0[:,1:]
        
        M = np.dot(P_cf0.T, np.dot(S, P_cf0))
        
        val, vec = sl.eigh(-M)
        U = np.dot(vec * 1.0 / np.sqrt(-val), vec.T)
        
        cf = np.dot(P_cf0, U)
        

        N, M = no-1,no-1
        U = np.eye(M)
        
        ic0 = 1
        # mf.mo_coeff=np.hstack((cf0[:,:ic0],cf))

        cascal=mf.CASSCF(8,8)
        cascal.kernel()
        dmcas=cascal.make_rdm1()
        # print(S)
        # print(dmcas.shape)
        # print(dmcas@S)

        rho_s=dmcas@S
        val,vec=sl.eig(rho_s)
        a = np.where(np.real(val) > 1.95)[0]
        print(np.real(vec[:,a]))
        print("+"*60)
        print(np.real(val[a]))
        # raise
        if VERBOSE > 1:
            print("Initial Orbitals:")
            print(cf)

    elif molecule <-20:

        "Dimer Trimer and so on calculations"
        if n == None:
            n = 2
        dis =3.0

        mol=gto.M()
        mol.atom=[]
        if molecule == -7:
            #mol.atom.append([10,(0.,0.,0.)])
            for i in range(n):
                mol.atom.append([10,(i*dis,0.,0.)])
        
        mol.unit='ang'
        if True:
            mol.basis={'Ne': gto.basis.parse('''
            Ne    S
                1.788000E+04           7.380000E-04          -1.720000E-04           0.000000E+00
                2.683000E+03           5.677000E-03          -1.357000E-03           0.000000E+00
                6.115000E+02           2.888300E-02          -6.737000E-03           0.000000E+00
                1.735000E+02           1.085400E-01          -2.766300E-02           0.000000E+00
                5.664000E+01           2.909070E-01          -7.620800E-02           0.000000E+00
                2.042000E+01           4.483240E-01          -1.752270E-01           0.000000E+00
                7.810000E+00           2.580260E-01          -1.070380E-01           0.000000E+00
                1.653000E+00           1.506300E-02           5.670500E-01           0.000000E+00
                4.869000E-01          -2.100000E-03           5.652160E-01           1.000000E+00
            Ne    P
                2.839000E+01           4.608700E-02           0.000000E+00
                6.270000E+00           2.401810E-01           0.000000E+00
                1.695000E+00           5.087440E-01           0.000000E+00
                4.317000E-01           4.556600E-01           1.000000E+00
            ''')}
        else:
            mol.basis={'Ne': gto.basis.parse('''
            Ne    S
                1.788000E+04           7.380000E-04          -1.720000E-04           0.000000E+00
                2.683000E+03           5.677000E-03          -1.357000E-03           0.000000E+00
                6.115000E+02           2.888300E-02          -6.737000E-03           0.000000E+00
                1.735000E+02           1.085400E-01          -2.766300E-02           0.000000E+00
                5.664000E+01           2.909070E-01          -7.620800E-02           0.000000E+00
                2.042000E+01           4.483240E-01          -1.752270E-01           0.000000E+00
                7.810000E+00           2.580260E-01          -1.070380E-01           0.000000E+00
                1.653000E+00           1.506300E-02           5.670500E-01           0.000000E+00
                4.869000E-01          -2.100000E-03           5.652160E-01           1.000000E+00
            Ne    P
                2.839000E+01           4.608700E-02           0.000000E+00
                6.270000E+00           2.401810E-01           0.000000E+00
                1.695000E+00           5.087440E-01           0.000000E+00
                4.317000E-01           4.556600E-01           1.000000E+00
            Ne     D
                2.202000E+00           1.0000000
            ''')}
        
        if VERBOSE:
            print(f"Using {mol.basis}")
        mol.build()

        mf=scf.RHF(mol).run()
        orb_mat=lo.Boys(mol).kernel(mf.mo_coeff[:,:mol.nelectron//2])
        print(orb_mat)


    return cf, U, cf0, S, ic0, N, M, mol, mf


def get_ints(molecule, mol, mf, cf, cf0, ic0):
    "Get all integrals using orbitals given"
    if molecule != -1:
        "2eri over active orbtials only"
        if VERBOSE > 1:
            print("Entering ints")
        eri = ao2mo.outcore.full_iofree(mol, cf, aosym="s1")
        no = cf.shape[1]
        eri = eri.reshape((no,) * 4)
        if VERBOSE > 1:
            print("eri done")

        if VERBOSE > 1:
            print("Calculating Hc")
        Hc = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
        enr = mol.energy_nuc()
        # print(mol.charge)
        if VERBOSE > 1:
            print("Hc done")

    "remove core potential from hamiltonian(ic0=number of core)"
    if molecule > 0 or molecule < -5:
        dm_core = np.dot(cf0[:, :ic0], cf0[:, :ic0].T)
        G = mf.get_veff(dm=dm_core * 2)
        ecore = 2 * np.einsum("ij,ji->", Hc, dm_core)
        ecore += np.einsum("ij,ji->", G, dm_core)
        
    else:
        nb = cf.shape[0]
        G = np.zeros(
            (
                nb,
                nb,
            )
        )
        ecore = 0.0

    i1s = np.dot(cf.T, np.dot(Hc + G, cf))

    return i1s, eri, enr, ecore


def write_ints(i1s, eri, U, enr, ecore):
    """Write integrals for use in Carlos' code,
    Adjust path as needed"""
    from sys import path

    path.append("/zfshomes/jrkeyes/cluster_fool/cluster")
    from write_ints import write_ints_
    import save_det

    no = U.shape[0]
    write_ints_("tmp.ints", enr + ecore, i1s, eri)
    save_det.write_ruhf("tmp.det", np.eye(no), U)


def expand_ints(cf, i1s, eri, M):
    """Expand restricted integral and coefficients to GHF form,
    eri will be changed from chemist to physics notation"""
    
    cf_ = np.zeros((cf.shape[0], 2 * M))
    cf_[:, ::2] = cf
    cf_[:, 1::2] = cf

    i1s_ = np.zeros((2 * M,) * 2)
    i1s_[::2, ::2] = i1s
    i1s_[1::2, 1::2] = i1s

    eri = eri.transpose(0, 2, 1, 3)
    eri_ = np.zeros((2 * M,) * 4)
    eri_[::2, ::2, ::2, ::2] = eri
    eri_[1::2, 1::2, 1::2, 1::2] = eri
    eri_[1::2, ::2, 1::2, ::2] = eri
    eri_[::2, 1::2, ::2, 1::2] = eri

    if VERBOSE > 3:
        print("One electron Hamiltonian:")
        print(i1s)
        print("Two Electron Integrals:")
        print(eri)

    return cf_, i1s_, eri_


def Build_Cmap(Norb, nclus, nup, ndn):
    """
    Build the Cmap tensor connecting fermionic and cluster operators,
    states will be ordered by particle number and Sz.
    nup and ndn are the number of particles in the initial guess.
    Assumes all clusters are the same(generalize later)
    """
    from itertools import combinations

    Cmap = np.zeros((2 ** (2 * Norb), 2 ** (2 * Norb), 2 * Norb, nclus))

    "Construct states as outer products of up and down states"

    # Build state vectors
    Cup = np.zeros((Norb, 2**Norb))
    k = 0
    for b in combinations(np.arange(Norb), nup):
        Cup[b, k] = 1
        k += 1
    for i in range(Norb + 1):
        if i == nup:
            continue
        for b in combinations(np.arange(Norb), i):
            Cup[b, k] = 1
            k += 1
    part_up = np.einsum("pq->q", Cup)

    Cdn = np.zeros((Norb, 2**Norb))
    k = 0
    for b in combinations(np.arange(Norb), ndn):
        Cdn[b, k] = 1
        k += 1
    for i in range(Norb + 1):
        if i == ndn:
            continue
        for b in combinations(np.arange(Norb), i):
            Cdn[b, k] = 1
            k += 1
    part_dn = np.einsum("pq->q", Cdn)

    # build anihilation operators
    for i in range(Norb):
        # up first
        up_map = np.zeros((2**Norb, 2**Norb))
        for j in range(2**Norb):
            if Cup[i, j]:
                a = Cup[:, j].copy()
                a[i] = 0
                for k in range(2**Norb):
                    if np.all(a == Cup[:, k]):
                        up_map[k, j] = np.power(-1, np.sum(Cup[i + 1 :, j]))

        dn_map = np.zeros((2**Norb, 2**Norb))
        for j in range(2**Norb):
            #            dn_map[j,j] = np.power(-1,np.sum(Cdn[i:,j]))
            dn_map[j, j] = np.power(-1, np.sum(Cdn[:, j]))

        Cmap[:, :, 2 * i, 0] = np.einsum("pq,rs->prqs", up_map, dn_map).reshape(
            (2 ** (2 * Norb),) * 2
        )

        # down next
        dn_map = np.zeros((2**Norb, 2**Norb))
        for j in range(2**Norb):
            if Cdn[i, j]:
                a = Cdn[:, j].copy()
                a[i] = 0
                for k in range(2**Norb):
                    if np.all(a == Cdn[:, k]):
                        dn_map[k, j] = np.power(-1, np.sum(Cdn[i + 1 :, j]))

        #        up_map = np.zeros((2**Norb,2**Norb))
        #        for j in range(2**Norb):
        #            up_map[j,j] = np.power(-1,np.sum(Cup[i+1:,j]))
        up_map = np.eye(2**Norb)

        Cmap[:, :, 2 * i + 1, 0] = np.einsum("pq,rs->prqs", up_map, dn_map).reshape(
            (2 ** (2 * Norb),) * 2
        )

    # Reorder states by particle number then Sz
    part = np.add.outer(part_up, part_dn).flatten()
    Sz = np.subtract.outer(part_up, part_dn).flatten()

    temp = np.einsum("icp,kiq->ckpq", Cmap[:, :, :, 0], Cmap[:, :, :, 0], optimize=True)

    "order by particle number"
    ind = list(np.where(part == nup + ndn)[0])
    a = [0, len(ind)]
    for i in range(2 * Norb + 1):
        if i == nup + ndn:
            continue
        b = list(np.where(part == i)[0])
        a.append(len(b) + a[-1])
        ind += b

    part = part[ind]
    Sz = Sz[ind]
    for i in range(Norb * 2):
        Cmap[:, :, i, 0] = Cmap[:, :, i, 0][np.ix_(ind, ind)]

    "order by Sz within particle blocks"
    ind = []
    for i in range(len(a) - 1):
        p = int(part[a[i]])
        ind1 = []
        ind2 = []
        for z in range(-p, p + 1, 2):
            if z < 0:
                ind1 += list(np.where(Sz[a[i] : a[i + 1]] == z)[0])
            else:
                ind2 += list(np.where(Sz[a[i] : a[i + 1]] == z)[0])
        ind2 += ind1
        ind2 = list(np.array(ind2) + a[i])
        ind += ind2

    part = part[ind]
    Sz = Sz[ind]
    for i in range(Norb * 2):
        Cmap[:, :, i, 0] = Cmap[:, :, i, 0][np.ix_(ind, ind)]

    "Copy Cmap for other clusters"
    for w in range(1, nclus):
        Cmap[:, :, :, w] = Cmap[:, :, :, 0].copy()

    if VERBOSE > 1:
        print("Particle numbers:")
        print(part)
        print("Sz values:")
        print(Sz)
    if VERBOSE > 3:
        print("Cmap:")
        print(Cmap)

    return Cmap, part, Sz


def prepare_cMF(i1s, eri, Norb, nclus, nstat, anti):

    "A few preperatory steps for cMF"

    "Get Cmap"
    if VERBOSE > 1:
        print("Building Cmap")
    Cmap, part, Sz = Build_Cmap(Norb // 2, nclus, Norb // 4, Norb // 4)
    if VERBOSE > 1:
        print("Built Cmap")

    # Build initial guess(HF initial guess to start)
    if VERBOSE > 1:
        print("Building Initial Guess")
    D = np.zeros((nclus, nstat, nstat))
    D[0, :, :] = np.eye(nstat)
    for w in range(1, nclus):
        D[w] = D[0]
    if VERBOSE > 1:
        print("Built guess")
    if VERBOSE > 2:
        print("Initial guess:")
        print(D)
    
    # Transform integrals to config basis
    i1s = i1s.reshape((nclus, Norb) * 2).transpose(0, 2, 1, 3)
    eri = eri.reshape(
        (
            nclus,
            Norb,
        )
        * 4
    ).transpose(0, 2, 4, 6, 1, 3, 5, 7)
    if VERBOSE > 1:
        print("transformed integrals")

    if anti:
        eri -= eri.transpose(0, 1, 3, 2, 4, 5, 7, 6)

    return Cmap, part, Sz, D, i1s, eri


def build_h0(i1s, eri, D, Cmap, Norb, nclus, nstat, anti):
    """Build the cMF H0 operator given integrals and cluster states
    Outputs the single cluster and two cluster terms separately, returns density"""
    # Build density
    den = np.einsum("wi,wk->wik", D[:, :, 0], D[:, :, 0], optimize=True)

    H1c = np.zeros((nclus, nstat, nstat))
    H2c = np.zeros((nclus, nstat, nstat))

    # Add core ham contribution
    for i in range(nclus):
        temp = np.einsum(
            "pq,jip->qij", i1s[i, i, :, :], Cmap[:, :, :, i], optimize=True
        )
        H1c[i] += np.einsum("qij,jlq->il", temp, Cmap[:, :, :, i], optimize=True)

    # Add single cluster 2 electron term
    for i in range(nclus):
        temp = np.einsum(
            "pqrs,jip->qrsij", eri[i, i, i, i], Cmap[:, :, :, i], optimize=True
        )
        temp2 = np.einsum("qrsij,kjq->rsik", temp, Cmap[:, :, :, i], optimize=True)
        temp = np.einsum("rsik,kls->ril", temp2, Cmap[:, :, :, i], optimize=True)
        if not anti:
            H1c[i] += 0.5 * np.einsum(
                "ril,lmr->im", temp, Cmap[:, :, :, i], optimize=True
            )
        else:
            H1c[i] += 0.25 * np.einsum(
                "ril,lmr->im", temp, Cmap[:, :, :, i], optimize=True
            )

    # Add two cluster interaction
    for w in range(nclus):
        for x in range(nclus):
            if w == x:
                continue
            temp = np.einsum(
                "pqrs,mlq->prslm", eri[w, x, w, x], Cmap[:, :, :, x], optimize=True
            )
            temp2 = np.einsum("prslm,mns->prln", temp, Cmap[:, :, :, x], optimize=True)
            temp = np.einsum("prln,nl->pr", temp2, den[x], optimize=True)
            temp2 = np.einsum("pr,jip->rij", temp, Cmap[:, :, :, w], optimize=True)
            H2c[w, :, :] += np.einsum(
                "rij,jkr->ik", temp2, Cmap[:, :, :, w], optimize=True
            )
            if not anti:
                temp = np.einsum(
                    "pqsr,mlq->prslm", eri[w, x, x, w], Cmap[:, :, :, x], optimize=True
                )
                temp2 = np.einsum(
                    "prslm,mns->prln", temp, Cmap[:, :, :, x], optimize=True
                )
                temp = np.einsum("prln,nl->pr", temp2, den[x], optimize=True)
                temp2 = np.einsum("pr,jip->rij", temp, Cmap[:, :, :, w], optimize=True)
                H2c[w, :, :] -= np.einsum(
                    "rij,jkr->ik", temp2, Cmap[:, :, :, w], optimize=True
                )

    return H1c, H2c, den


def MF_Energy(H1c, H2c, den, nclus):
    "Calculates cMF Energy from H0 and density"
    Energy = 0.0
    for w in range(nclus):
        Energy += np.trace((H1c[w] + 0.5 * H2c[w]).dot(den[w]))
    return Energy


def cMF_state(i1s, eri, D, Cmap, Norb, nclus, nstat, anti, nstep=32, thrsh=1.0e-8):
    """Optimize the cMF state vectors. Currently with repeated diagonalization. Implement DIIS later.
    Returns energy ,state matrix, and cluster energies"""
    status = False
    E_old = 834789243.0
    
    for n in range(nstep):
        
        if VERBOSE > 3:
            print(f"cMF opt step {n}")
        # Get H0
        H1c, H2c, den = build_h0(i1s, eri, D, Cmap, Norb, nclus, nstat, anti)
        Energy = MF_Energy(H1c, H2c, den, nclus)
        
        # Check for convergence
        if np.abs(E_old - Energy) < thrsh:
            status = True
            break
        elif n == nstep - 1:
            print("Failed to converge state")
            raise
        E_old = Energy

        # Diagonalize H0
        orben = np.zeros((nclus, nstat))
        for i in range(nclus):
            val, vec = sl.eigh(H1c[i, :, :] + H2c[i, :, :])

            "Make sure ground state has correct number of particles"
            a = np.where(vec[0, :] != 0)[0][0]

            #hack
            # a = np.where(vec[0]==max(vec[0,:],key=abs))[0][0]
            # print(a)
            if a:
                vec[:, 0], vec[:, a] = vec[:, a].copy(), vec[:, 0].copy()
                val[0], val[a] = val[a].copy(), val[0].copy()
            D[i] = vec.copy()

            orben[i] = val.copy()
            if i==0:
                # print(vec[:4,:])
                print(Energy)
                

        # print(D[0,:,:4])
        
        
        if VERBOSE > 3:
            print(f"Current cMF energy: {Energy}")
        if VERBOSE > 4:
            print("Current H0:")
            print(H1c + H2c)
            print("Current orbitals:")
            print(D)
            print("Current Orbital Energies:")
            print(orben)

    if status:
        return Energy, D, orben


def update_orbs(C, Z):
    """return e^(Z)C"""
    if np.any(Z[::2, 1::2]) or np.any(Z[1::2, ::2]):
        raise Exception("Attempting to mix orbitals of different spin")
    U = sl.expm(Z)
    C_ = C.dot(U)
    return C_


def num_deriv(C, i1s, eri, D, Cmap, Norb, nclus, nstat, anti, S, M):
    """Calculate the numerical orbital derivative"""
    alpha = 0.000001
    n = Norb * nclus
    G = np.zeros((n,) * 2)
    for i in range(nclus):
        for j in range(i + 1, nclus):
            g = np.zeros((Norb, Norb))
            print(i, j)
            for a in range(Norb):
                for b in range(Norb):
                    if a % 2 != b % 2:
                        continue

                    n = Norb * nclus
                    Z = np.zeros((n, n))
                    for k in range(nclus):
                        for l in range(k + 1, nclus):
                            if k == i and l == j:
                                Z[Norb * k + a, Norb * l + b] += alpha
                    Z -= Z.T
                    C_ = update_orbs(C, Z)
                    i1s_, eri_ = update_ints(C, C_, i1s, eri, S, M, nclus, Norb)
                    H1c, H2c, den = build_h0(
                        i1s_, eri_, D, Cmap, Norb, nclus, nstat, anti
                    )
                    Energy = MF_Energy(H1c, H2c, den, nclus)
                    g[a, b] += Energy

                    n = Norb * nclus
                    Z = np.zeros((n, n))
                    for k in range(nclus):
                        for l in range(k + 1, nclus):
                            if k == i and l == j:
                                Z[Norb * k + a, Norb * l + b] -= alpha
                    Z -= Z.T
                    C_ = update_orbs(C, Z)
                    i1s_, eri_ = update_ints(C, C_, i1s, eri, S, M, nclus, Norb)
                    H1c, H2c, den = build_h0(
                        i1s_, eri_, D, Cmap, Norb, nclus, nstat, anti
                    )
                    Energy = MF_Energy(H1c, H2c, den, nclus)
                    g[a, b] -= Energy
                    g[a, b] /= 2 * alpha
            G[Norb * i : Norb * (i + 1), Norb * j : Norb * (j + 1)] = g.copy()
    G -= G.T
    return G


def update_ints(C1, C2, i1s, eri, S, M, nclus, Norb):
    """Convert integrals stored in the C1 basis to the C2 basis"""
    C = np.zeros((2 * M,) * 2)
    C[::2, ::2] = C1[:, ::2].T.dot(S).dot(C2[:, ::2])
    C[1::2, 1::2] = C1[:, 1::2].T.dot(S).dot(C2[:, 1::2])
    i1s = i1s.transpose(0, 2, 1, 3).reshape((2 * M,) * 2)
    eri = eri.transpose(0, 4, 1, 5, 2, 6, 3, 7).reshape((2 * M,) * 4)
    i1s = np.einsum("pq,pi,qj->ij", i1s, C, C, optimize=True)
    eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, C, C, C, C, optimize=True)
    i1s = i1s.reshape(
        (
            nclus,
            Norb,
        )
        * 2
    ).transpose(0, 2, 1, 3)
    eri = eri.reshape(
        (
            nclus,
            Norb,
        )
        * 4
    ).transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return i1s, eri


def cMF_orb_grad_nm(n, m, i1s, eri, D, Cmap, Norb, nclus, anti):
    """Calculate the orbital rotation gradient at z=0 between cluster n and m.
    Assume all clusters have same number of orbs, generalize later"""
    g = np.zeros((Norb, Norb))
    if n == m:
        return g
    # Build density
    den = np.einsum("wi,wk->wik", D[:, :, 0], D[:, :, 0], optimize=True)
    # Core term
    temp = np.einsum("bq,jkq->jkb", i1s[n, m, :, :], Cmap[:, :, :, m], optimize=True)
    temp2 = np.einsum("jkb,ik->jib", temp, den[m, :, :], optimize=True)
    g += np.einsum("jib,jia->ab", temp2, Cmap[:, :, :, m], optimize=True)

    temp = np.einsum("pa,jip->ija", i1s[n, m, :, :], Cmap[:, :, :, n], optimize=True)
    temp2 = np.einsum("ija,ik->kja", temp, den[n, :, :], optimize=True)
    g -= np.einsum("kja,jkb->ab", temp2, Cmap[:, :, :, n], optimize=True)

    # Two cluster eri contributions(optimize later)
    temp = np.einsum(
        "bqrs,lzr->lzbqs", eri[n, m, m, m], Cmap[:, :, :, m], optimize=True
    )
    temp2 = np.einsum("lzbqs,kls->kzbq", temp, Cmap[:, :, :, m], optimize=True)
    temp = np.einsum("kzbq,kjq->jzb", temp2, Cmap[:, :, :, m], optimize=True)
    temp2 = np.einsum("jzb,jia->izab", temp, Cmap[:, :, :, m], optimize=True)
    g += 0.5 * np.einsum("izab,iz->ab", temp2, den[m, :, :], optimize=True)

    if not anti:
        temp = np.einsum(
            "pbrs,lzr->lzbps", eri[m, n, m, m], Cmap[:, :, :, m], optimize=True
        )
        temp2 = np.einsum("lzbps,kls->kzbp", temp, Cmap[:, :, :, m], optimize=True)
        temp = np.einsum("kzbp,kjp->jzb", temp2, Cmap[:, :, :, m], optimize=True)
        temp2 = np.einsum("jzb,jia->izab", temp, Cmap[:, :, :, m], optimize=True)
        g -= 0.5 * np.einsum("izab,iz->ab", temp2, den[m, :, :], optimize=True)

    temp = np.einsum(
        "pqas,jip->ijqas", eri[n, n, m, n], Cmap[:, :, :, n], optimize=True
    )
    temp2 = np.einsum("ijqas,kjq->ikas", temp, Cmap[:, :, :, n], optimize=True)
    temp = np.einsum("ikas,kjs->ija", temp2, Cmap[:, :, :, n], optimize=True)
    temp2 = np.einsum("ija,jkb->ikab", temp, Cmap[:, :, :, n], optimize=True)
    g -= 0.5 * np.einsum("ikab,ik->ab", temp2, den[n, :, :], optimize=True)

    if not anti:
        temp = np.einsum(
            "pqra,jip->ijqar", eri[n, n, n, m], Cmap[:, :, :, n], optimize=True
        )
        temp2 = np.einsum("ijqar,kjq->ikar", temp, Cmap[:, :, :, n], optimize=True)
        temp = np.einsum("ikar,kjr->ija", temp2, Cmap[:, :, :, n], optimize=True)
        temp2 = np.einsum("ija,jkb->ikab", temp, Cmap[:, :, :, n], optimize=True)
        g += 0.5 * np.einsum("ikab,ik->ab", temp2, den[n, :, :], optimize=True)

    # Three cluster contribution(optimize later)
    for o in range(nclus):
        if o != m:
            temp = np.einsum(
                "bqrs,jiq->ijbrs", eri[n, o, m, o], Cmap[:, :, :, o], optimize=True
            )
            temp2 = np.einsum("ijbrs,jks->ikbr", temp, Cmap[:, :, :, o], optimize=True)
            temp = np.einsum("ikbr,ik->br", temp2, den[o, :, :], optimize=True)
            temp2 = np.einsum("br,jkr->jkb", temp, Cmap[:, :, :, m], optimize=True)
            temp = np.einsum("jkb,jia->ikab", temp2, Cmap[:, :, :, m], optimize=True)
            g += 0.5 * np.einsum("ikab,ik->ab", temp, den[m, :, :], optimize=True)

            temp = np.einsum(
                "bqrs,jiq->ijbrs", eri[n, o, o, m], Cmap[:, :, :, o], optimize=True
            )
            temp2 = np.einsum("ijbrs,jkr->ikbs", temp, Cmap[:, :, :, o], optimize=True)
            temp = np.einsum("ikbs,ik->bs", temp2, den[o, :, :], optimize=True)
            temp2 = np.einsum("bs,jks->jkb", temp, Cmap[:, :, :, m], optimize=True)
            temp = np.einsum("jkb,jia->ikab", temp2, Cmap[:, :, :, m], optimize=True)
            g -= 0.5 * np.einsum("ikab,ik->ab", temp, den[m, :, :], optimize=True)

            if not anti:
                temp = np.einsum(
                    "pbrs,jip->ijbrs", eri[o, n, m, o], Cmap[:, :, :, o], optimize=True
                )
                temp2 = np.einsum(
                    "ijbrs,jks->ikbr", temp, Cmap[:, :, :, o], optimize=True
                )
                temp = np.einsum("ikbr,ik->br", temp2, den[o, :, :], optimize=True)
                temp2 = np.einsum("br,jkr->jkb", temp, Cmap[:, :, :, m], optimize=True)
                temp = np.einsum(
                    "jkb,jia->ikab", temp2, Cmap[:, :, :, m], optimize=True
                )
                g -= 0.5 * np.einsum("ikab,ik->ab", temp, den[m, :, :], optimize=True)

            if not anti:
                temp = np.einsum(
                    "pbrs,jip->ijbrs", eri[o, n, o, m], Cmap[:, :, :, o], optimize=True
                )
                temp2 = np.einsum(
                    "ijbrs,jkr->ikbs", temp, Cmap[:, :, :, o], optimize=True
                )
                temp = np.einsum("ikbs,ik->bs", temp2, den[o, :, :], optimize=True)
                temp2 = np.einsum("bs,jks->jkb", temp, Cmap[:, :, :, m], optimize=True)
                temp = np.einsum(
                    "jkb,jia->ikab", temp2, Cmap[:, :, :, m], optimize=True
                )
                g += 0.5 * np.einsum("ikab,ik->ab", temp, den[m, :, :], optimize=True)

        if o != n:
            temp = np.einsum(
                "pqas,jiq->ijpas", eri[n, o, m, o], Cmap[:, :, :, o], optimize=True
            )
            temp2 = np.einsum("ijpas,jks->ikpa", temp, Cmap[:, :, :, o], optimize=True)
            temp = np.einsum("ikpa,ik->pa", temp2, den[o, :, :], optimize=True)
            temp2 = np.einsum("pa,jip->ija", temp, Cmap[:, :, :, n], optimize=True)
            temp = np.einsum("ija,jkb->ikab", temp2, Cmap[:, :, :, n], optimize=True)
            g -= 0.5 * np.einsum("ikab,ik->ab", temp, den[n, :, :], optimize=True)

            temp = np.einsum(
                "pqas,jip->ijqas", eri[o, n, m, o], Cmap[:, :, :, o], optimize=True
            )
            temp2 = np.einsum("ijqas,jks->ikqa", temp, Cmap[:, :, :, o], optimize=True)
            temp = np.einsum("ikqa,ik->qa", temp2, den[o, :, :], optimize=True)
            temp2 = np.einsum("qa,jiq->ija", temp, Cmap[:, :, :, n], optimize=True)
            temp = np.einsum("ija,jkb->ikab", temp2, Cmap[:, :, :, n], optimize=True)
            g += 0.5 * np.einsum("ikab,ik->ab", temp, den[n, :, :], optimize=True)

            if not anti:
                temp = np.einsum(
                    "pqra,jiq->ijpar", eri[n, o, o, m], Cmap[:, :, :, o], optimize=True
                )
                temp2 = np.einsum(
                    "ijpar,jkr->ikpa", temp, Cmap[:, :, :, o], optimize=True
                )
                temp = np.einsum("ikpa,ik->pa", temp2, den[o, :, :], optimize=True)
                temp2 = np.einsum("pa,jip->ija", temp, Cmap[:, :, :, n], optimize=True)
                temp = np.einsum(
                    "ija,jkb->ikab", temp2, Cmap[:, :, :, n], optimize=True
                )
                g += 0.5 * np.einsum("ikab,ik->ab", temp, den[n, :, :], optimize=True)

            if not anti:
                temp = np.einsum(
                    "pqra,jip->ijqar", eri[o, n, o, m], Cmap[:, :, :, o], optimize=True
                )
                temp2 = np.einsum(
                    "ijqar,jkr->ikqa", temp, Cmap[:, :, :, o], optimize=True
                )
                temp = np.einsum("ikqa,ik->qa", temp2, den[o, :, :], optimize=True)
                temp2 = np.einsum("qa,jiq->ija", temp, Cmap[:, :, :, n], optimize=True)
                temp = np.einsum(
                    "ija,jkb->ikab", temp2, Cmap[:, :, :, n], optimize=True
                )
                g -= 0.5 * np.einsum("ikab,ik->ab", temp, den[n, :, :], optimize=True)
    return 2 * g.T


def cMF_orb_grad(i1s, eri, D, Cmap, nclus, Norb, anti):
    """Calculate the orbital rotation gradient at z=0 w/ unitary thouless"""
    n = Norb * nclus
    G = np.zeros((n, n))
    for i in range(nclus):
        for j in range(i + 1, nclus):
            g = cMF_orb_grad_nm(i, j, i1s, eri, D, Cmap, Norb, nclus, anti)
            G[Norb * i : Norb * (i + 1), Norb * j : Norb * (j + 1)] = g.copy()
    G -= G.T

    return G


def line_search(G, i1s, eri, C, E0, D, nstep_state, thrsh_state):
    """Performs simple linesearch for the orbital gradient, not currently working"""
    alpha = 0.1
    beta = 0.1
    step = 0.1
    G_ = [1, 1, 1]
    while True:
        for i in range(len(G)):
            G_[i] = -G[i] * alpha
        C_ = update_orbs(C, G_)
        i1s, eri = update_ints(C, C_, i1s, eri)
        Energy, D, orben = cMF_state(i1s, eri, D, nstep=nstep_state, thrsh=thrsh_state)
        if Energy > E0:
            break
        else:
            E0 = Energy
            alpha += step
    return alpha


def num_hess(C, i1s, eri, D, Cmap, Norb, nclus, nstat, anti, S, M):
    """Calculate the numerical orbital hessian"""
    alpha = 0.0001
    n = Norb * nclus
    H = np.zeros((n,) * 4)
    for i in range(nclus):
        for j in range(i + 1, nclus):
            for ii in range(nclus):
                for jj in range(ii + 1, nclus):
                    h = np.zeros((Norb, Norb, Norb, Norb))
                    print(i, j, ii, jj)
                    for a in range(Norb):
                        for b in range(Norb):
                            if a % 2 != b % 2:
                                continue
                            for c in range(Norb):
                                for d in range(Norb):
                                    if c % 2 != d % 2:
                                        continue
                                    n = Norb * nclus
                                    Z = np.zeros((n, n))
                                    for k in range(nclus):
                                        for l in range(k + 1, nclus):
                                            if k == i and l == j:
                                                Z[Norb * k + a, Norb * l + b] += alpha
                                            if k == ii and l == jj:
                                                Z[Norb * k + c, Norb * l + d] += alpha
                                    Z -= Z.T
                                    C_ = update_orbs(C, Z)
                                    i1s_, eri_ = update_ints(
                                        C, C_, i1s, eri, S, M, nclus, Norb
                                    )
                                    H1c, H2c, den = build_h0(
                                        i1s_, eri_, D, Cmap, Norb, nclus, nstat, anti
                                    )
                                    Energy = MF_Energy(H1c, H2c, den, nclus)
                                    h[a, b, c, d] += Energy

                                    n = Norb * nclus
                                    Z = np.zeros((n, n))
                                    for k in range(nclus):
                                        for l in range(k + 1, nclus):
                                            if k == i and l == j:
                                                Z[Norb * k + a, Norb * l + b] += alpha
                                            if k == ii and l == jj:
                                                Z[Norb * k + c, Norb * l + d] -= alpha
                                    Z -= Z.T
                                    C_ = update_orbs(C, Z)
                                    i1s_, eri_ = update_ints(
                                        C, C_, i1s, eri, S, M, nclus, Norb
                                    )
                                    H1c, H2c, den = build_h0(
                                        i1s_, eri_, D, Cmap, Norb, nclus, nstat, anti
                                    )
                                    Energy = MF_Energy(H1c, H2c, den, nclus)
                                    h[a, b, c, d] -= Energy

                                    n = Norb * nclus
                                    Z = np.zeros((n, n))
                                    for k in range(nclus):
                                        for l in range(k + 1, nclus):
                                            if k == i and l == j:
                                                Z[Norb * k + a, Norb * l + b] -= alpha
                                            if k == ii and l == jj:
                                                Z[Norb * k + c, Norb * l + d] += alpha
                                    Z -= Z.T
                                    C_ = update_orbs(C, Z)
                                    i1s_, eri_ = update_ints(
                                        C, C_, i1s, eri, S, M, nclus, Norb
                                    )
                                    H1c, H2c, den = build_h0(
                                        i1s_, eri_, D, Cmap, Norb, nclus, nstat, anti
                                    )
                                    Energy = MF_Energy(H1c, H2c, den, nclus)
                                    h[a, b, c, d] -= Energy

                                    n = Norb * nclus
                                    Z = np.zeros((n, n))
                                    for k in range(nclus):
                                        for l in range(k + 1, nclus):
                                            if k == i and l == j:
                                                Z[Norb * k + a, Norb * l + b] -= alpha
                                            if k == ii and l == jj:
                                                Z[Norb * k + c, Norb * l + d] -= alpha
                                    Z -= Z.T
                                    C_ = update_orbs(C, Z)
                                    i1s_, eri_ = update_ints(
                                        C, C_, i1s, eri, S, M, nclus, Norb
                                    )
                                    H1c, H2c, den = build_h0(
                                        i1s_, eri_, D, Cmap, Norb, nclus, nstat, anti
                                    )
                                    Energy = MF_Energy(H1c, H2c, den, nclus)
                                    h[a, b, c, d] += Energy

                    h /= 4 * alpha * alpha
                    H[
                        Norb * i : Norb * (i + 1),
                        Norb * j : Norb * (j + 1),
                        Norb * ii : Norb * (ii + 1),
                        Norb * jj : Norb * (jj + 1),
                    ] = h.copy()
                    H[
                        Norb * i : Norb * (i + 1),
                        Norb * j : Norb * (j + 1),
                        Norb * ii : Norb * (jj + 1),
                        Norb * ii : Norb * (ii + 1),
                    ] = -h.transpose(0, 1, 3, 2)
                    H[
                        Norb * j : Norb * (j + 1),
                        Norb * i : Norb * (i + 1),
                        Norb * jj : Norb * (ii + 1),
                        Norb * jj : Norb * (jj + 1),
                    ] = -h.transpose(1, 0, 2, 3)
                    H[
                        Norb * j : Norb * (j + 1),
                        Norb * i : Norb * (i + 1),
                        Norb * ii : Norb * (jj + 1),
                        Norb * ii : Norb * (ii + 1),
                    ] = h.transpose(1, 0, 3, 2)
    return H


def cMF_orb_hess_ijkl(i, j, k, l, i1s, eri, Cmap, Norb, nclus, nstat):
    """Calculate [[H,a^!b],c^!d] between cluster i and j then k and l.
    Assume all clusters have same number of orbs, generalize later"""

    h1 = np.zeros((Norb,) * 4)
    h2 = np.zeros((Norb,) * 4)
    # Core term
    if j == k:
        temp = np.einsum("jp,jd->pd", Cmap[:, 0, :, l], Cmap[:, 0, :, l], optimize=True)
        temp = np.einsum("pd,pa->ad", temp, i1s[l, i], optimize=True)
        h1 += np.einsum("ad,bc->abcd", temp, np.eye(Norb), optimize=True)

        temp = np.einsum("jc,jb->cb", Cmap[:, 0, :, k], Cmap[:, 0, :, j], optimize=True)
        h1 -= np.einsum("cb,da->abcd", temp, i1s[l, i], optimize=True)

    if i == l:
        temp = np.einsum("ja,jd->ad", Cmap[:, 0, :, i], Cmap[:, 0, :, l], optimize=True)
        h1 -= np.einsum("ad,bc->abcd", temp, i1s[j, k], optimize=True)

        temp = np.einsum("jc,jq->cq", Cmap[:, 0, :, k], Cmap[:, 0, :, k], optimize=True)
        temp = np.einsum("cq,bq->bc", temp, i1s[j, k], optimize=True)
        h1 += np.einsum("bc,ad->abcd", temp, np.eye(Norb), optimize=True)

    # Single cluster eri
    if j == k:
        temp = np.einsum(
            "pqas,kp->kqas", eri[l, l, i, l], Cmap[:, 0, :, l], optimize=True
        )
        temp = np.einsum("kqas,jkq->jas", temp, Cmap[:, :, :, l], optimize=True)
        temp = np.einsum("jas,jks->ka", temp, Cmap[:, :, :, l], optimize=True)
        temp = np.einsum("ka,kd->ad", temp, Cmap[:, 0, :, l], optimize=True)
        h2 += np.einsum("ad,bc->abcd", temp, np.eye(Norb), optimize=True)

        temp = np.einsum(
            "pdas,kp->ksda", eri[k, l, i, k], Cmap[:, 0, :, k], optimize=True
        )
        temp = np.einsum("ksda,jkc->jscda", temp, Cmap[:, :, :, k], optimize=True)
        temp = np.einsum("jscda,jks->kcda", temp, Cmap[:, :, :, k], optimize=True)
        h2 -= np.einsum("kcda,kb->abcd", temp, Cmap[:, 0, :, k], optimize=True)

        temp = np.einsum(
            "dqas,kc->kqsadc", eri[l, k, i, k], Cmap[:, 0, :, k], optimize=True
        )
        temp = np.einsum("kqsadc,jkq->jsadc", temp, Cmap[:, :, :, k], optimize=True)
        temp = np.einsum("jsadc,jks->kadc", temp, Cmap[:, :, :, k], optimize=True)
        h2 -= np.einsum("kadc,kb->abcd", temp, Cmap[:, 0, :, k], optimize=True)

    if j == l:
        temp = np.einsum(
            "pqac,kp->kqac", eri[l, l, i, k], Cmap[:, 0, :, l], optimize=True
        )
        temp = np.einsum("kqac,jkq->jac", temp, Cmap[:, :, :, l], optimize=True)
        temp = np.einsum("jac,jkd->kacd", temp, Cmap[:, :, :, l], optimize=True)
        h2 += np.einsum("kacd,kb->abcd", temp, Cmap[:, 0, :, l], optimize=True)

    if i == l:
        temp = np.einsum(
            "pbcs,kd->kpsbcd", eri[l, j, k, l], Cmap[:, 0, :, l], optimize=True
        )
        temp = np.einsum("kpsbcd,jks->jpbcd", temp, Cmap[:, :, :, l], optimize=True)
        temp = np.einsum("jpbcd,jka->kpabcd", temp, Cmap[:, :, :, l], optimize=True)
        h2 -= np.einsum("kpabcd,kp->abcd", temp, Cmap[:, 0, :, l], optimize=True)

        temp = np.einsum(
            "pbrc,kr->kpbc", eri[l, j, l, k], Cmap[:, 0, :, l], optimize=True
        )
        temp = np.einsum("kpbc,jkd->jpbcd", temp, Cmap[:, :, :, l], optimize=True)
        temp = np.einsum("jpbcd,jka->kpabcd", temp, Cmap[:, :, :, l], optimize=True)
        h2 -= np.einsum("kpabcd,kp->abcd", temp, Cmap[:, 0, :, l], optimize=True)

        temp = np.einsum(
            "pbrs,kr->kpsb", eri[k, j, k, k], Cmap[:, 0, :, k], optimize=True
        )
        temp = np.einsum("kpsb,jks->jpb", temp, Cmap[:, :, :, k], optimize=True)
        temp = np.einsum("jpb,jkc->kpbc", temp, Cmap[:, :, :, k], optimize=True)
        temp = np.einsum("kpbc,kp->bc", temp, Cmap[:, 0, :, k], optimize=True)
        h2 += np.einsum("bc,ad->abcd", temp, np.eye(Norb), optimize=True)

    if i == k:
        temp = np.einsum(
            "dbrs,kr->ksbd", eri[l, j, k, k], Cmap[:, 0, :, k], optimize=True
        )
        temp = np.einsum("ksbd,jks->jbd", temp, Cmap[:, :, :, k], optimize=True)
        temp = np.einsum("jbd,jka->kabd", temp, Cmap[:, :, :, k], optimize=True)
        h2 += np.einsum("kabd,kc->abcd", temp, Cmap[:, 0, :, k], optimize=True)

    # Two cluster eri
    if j != l:
        temp = np.einsum(
            "pqac,kp->kqac", eri[j, l, i, k], Cmap[:, 0, :, j], optimize=True
        )
        temp = np.einsum("kqac,kb->qacb", temp, Cmap[:, 0, :, j], optimize=True)
        temp = np.einsum("qacb,kq->kacb", temp, Cmap[:, 0, :, l], optimize=True)
        h2 += 2 * np.einsum("kacb,kd->abcd", temp, Cmap[:, 0, :, l], optimize=True)

    if j != k:
        temp = np.einsum(
            "pdas,kp->kdas", eri[j, l, i, k], Cmap[:, 0, :, j], optimize=True
        )
        temp = np.einsum("kdas,kb->sabd", temp, Cmap[:, 0, :, j], optimize=True)
        temp = np.einsum("sabd,ks->kabd", temp, Cmap[:, 0, :, k], optimize=True)
        h2 -= np.einsum("kabd,kc->abcd", temp, Cmap[:, 0, :, k], optimize=True)

        temp = np.einsum(
            "dqas,kq->kdas", eri[l, j, i, k], Cmap[:, 0, :, j], optimize=True
        )
        temp = np.einsum("kdas,kb->sabd", temp, Cmap[:, 0, :, j], optimize=True)
        temp = np.einsum("sabd,ks->kabd", temp, Cmap[:, 0, :, k], optimize=True)
        h2 += np.einsum("kabd,kc->abcd", temp, Cmap[:, 0, :, k], optimize=True)

    if i != l:
        temp = np.einsum(
            "pbcs,kp->ksbc", eri[l, j, k, i], Cmap[:, 0, :, l], optimize=True
        )
        temp = np.einsum("ksbc,kd->sbcd", temp, Cmap[:, 0, :, l], optimize=True)
        temp = np.einsum("sbcd,ks->kbcd", temp, Cmap[:, 0, :, i], optimize=True)
        h2 -= np.einsum("kbcd,ka->abcd", temp, Cmap[:, 0, :, i], optimize=True)

        temp = np.einsum(
            "pbrc,kp->krbc", eri[l, j, i, k], Cmap[:, 0, :, l], optimize=True
        )
        temp = np.einsum("krbc,kd->rbcd", temp, Cmap[:, 0, :, l], optimize=True)
        temp = np.einsum("rbcd,kr->kbcd", temp, Cmap[:, 0, :, i], optimize=True)
        h2 += np.einsum("kbcd,ka->abcd", temp, Cmap[:, 0, :, i], optimize=True)

    if i != k:
        temp = np.einsum(
            "dbrs,kr->ksbd", eri[l, j, k, i], Cmap[:, 0, :, k], optimize=True
        )
        temp = np.einsum("ksbd,kc->sbcd", temp, Cmap[:, 0, :, k], optimize=True)
        temp = np.einsum("sbcd,ks->kbcd", temp, Cmap[:, 0, :, i], optimize=True)
        h2 += 2 * np.einsum("kbcd,ka->abcd", temp, Cmap[:, 0, :, i], optimize=True)

    for n in range(nclus):
        if n != l and j == k:
            temp = np.einsum(
                "pqas,kq->kpas", eri[l, n, i, n], Cmap[:, 0, :, n], optimize=True
            )
            temp = np.einsum("kpas,ks->pa", temp, Cmap[:, 0, :, n], optimize=True)
            temp = np.einsum("pa,kp->ka", temp, Cmap[:, 0, :, l], optimize=True)
            temp = np.einsum("ka,kd->ad", temp, Cmap[:, 0, :, l], optimize=True)
            h2 += 2 * np.einsum("ad,bc->abcd", temp, np.eye(Norb), optimize=True)

        if n != k and j == k:
            temp = np.einsum(
                "pdas,kp->ksda", eri[n, l, i, n], Cmap[:, 0, :, n], optimize=True
            )
            temp = np.einsum("ksda,ks->ad", temp, Cmap[:, 0, :, n], optimize=True)
            temp = np.einsum("da,kc->kcda", temp, Cmap[:, 0, :, k], optimize=True)
            h2 += np.einsum("kcda,kb->abcd", temp, Cmap[:, 0, :, k], optimize=True)

            temp = np.einsum(
                "dqas,kq->ksad", eri[l, n, i, n], Cmap[:, 0, :, n], optimize=True
            )
            temp = np.einsum("ksad,ks->ad", temp, Cmap[:, 0, :, n], optimize=True)
            temp = np.einsum("ad,kc->kadc", temp, Cmap[:, 0, :, k], optimize=True)
            h2 -= np.einsum("kadc,kb->abcd", temp, Cmap[:, 0, :, k], optimize=True)

        if n != k and i == l:
            temp = np.einsum(
                "pbrs,kp->krsb", eri[n, j, n, k], Cmap[:, 0, :, n], optimize=True
            )
            temp = np.einsum("krsb,kr->sb", temp, Cmap[:, 0, :, n], optimize=True)
            temp = np.einsum("sb,ks->kb", temp, Cmap[:, 0, :, k], optimize=True)
            temp = np.einsum("kb,kc->bc", temp, Cmap[:, 0, :, k], optimize=True)
            h2 += 2 * np.einsum("bc,ad->abcd", temp, np.eye(Norb), optimize=True)

        if n != i and i == l:
            temp = np.einsum(
                "pbcs,kp->ksbc", eri[n, j, k, n], Cmap[:, 0, :, n], optimize=True
            )
            temp = np.einsum("ksbc,ks->bc", temp, Cmap[:, 0, :, n], optimize=True)
            temp = np.einsum("bc,ka->kabc", temp, Cmap[:, 0, :, i], optimize=True)
            h2 += np.einsum("kabc,kd->abcd", temp, Cmap[:, 0, :, i], optimize=True)

            temp = np.einsum(
                "pbrc,kp->krbc", eri[n, j, n, k], Cmap[:, 0, :, n], optimize=True
            )
            temp = np.einsum("krbc,kr->bc", temp, Cmap[:, 0, :, n], optimize=True)
            temp = np.einsum("bc,ka->kabc", temp, Cmap[:, 0, :, i], optimize=True)
            h2 -= np.einsum("kabc,kd->abcd", temp, Cmap[:, 0, :, i], optimize=True)

    h1 /= 2
    h2 /= 4
    h = h1 + h2
    "Remove spin mixing"
    A = np.zeros((Norb, Norb))
    for i in range(Norb):
        for j in range(Norb):
            if i % 2 == j % 2:
                A[i, j] = A[j, i] = 1.0
    h = np.einsum("pqrs,pq,rs->pqrs", h, A, A, optimize=True)
    return h


def cMF_orb_hess(i1s, eri, Cmap, D, Norb, nclus, nstat, anti):
    """Calculate the orbital hessian"""

    if not anti:
        eri -= eri.transpose(0, 1, 3, 2, 4, 5, 7, 6)
    Cmap = np.einsum("ijpw,wik,wjl->klpw", Cmap, D, D, optimize=True)

    n = Norb * nclus
    H = np.zeros((n,) * 4)
    for i in range(nclus):
        for j in range(i + 1, nclus):
            for k in range(nclus):
                for l in range(k + 1, nclus):
                    h = np.zeros((Norb,) * 4)
                    h += cMF_orb_hess_ijkl(
                        i, j, k, l, i1s, eri, Cmap, Norb, nclus, nstat
                    )
                    h -= cMF_orb_hess_ijkl(
                        j, i, k, l, i1s, eri, Cmap, Norb, nclus, nstat
                    ).transpose(1, 0, 2, 3)
                    h -= cMF_orb_hess_ijkl(
                        i, j, l, k, i1s, eri, Cmap, Norb, nclus, nstat
                    ).transpose(0, 1, 3, 2)
                    h += cMF_orb_hess_ijkl(
                        j, i, l, k, i1s, eri, Cmap, Norb, nclus, nstat
                    ).transpose(1, 0, 3, 2)
                    h += cMF_orb_hess_ijkl(
                        k, l, i, j, i1s, eri, Cmap, Norb, nclus, nstat
                    ).transpose(2, 3, 0, 1)
                    h -= cMF_orb_hess_ijkl(
                        k, l, j, i, i1s, eri, Cmap, Norb, nclus, nstat
                    ).transpose(3, 2, 0, 1)
                    h -= cMF_orb_hess_ijkl(
                        l, k, i, j, i1s, eri, Cmap, Norb, nclus, nstat
                    ).transpose(2, 3, 1, 0)
                    h += cMF_orb_hess_ijkl(
                        l, k, j, i, i1s, eri, Cmap, Norb, nclus, nstat
                    ).transpose(3, 2, 1, 0)

                    H[
                        Norb * i : Norb * (i + 1),
                        Norb * j : Norb * (j + 1),
                        Norb * k : Norb * (k + 1),
                        Norb * l : Norb * (l + 1),
                    ] = h
                    H[
                        Norb * i : Norb * (i + 1),
                        Norb * j : Norb * (j + 1),
                        Norb * l : Norb * (l + 1),
                        Norb * k : Norb * (k + 1),
                    ] = -h.transpose(0, 1, 3, 2)
                    H[
                        Norb * j : Norb * (j + 1),
                        Norb * i : Norb * (i + 1),
                        Norb * k : Norb * (k + 1),
                        Norb * l : Norb * (l + 1),
                    ] = -h.transpose(1, 0, 2, 3)
                    H[
                        Norb * j : Norb * (j + 1),
                        Norb * i : Norb * (i + 1),
                        Norb * l : Norb * (l + 1),
                        Norb * k : Norb * (k + 1),
                    ] = h.transpose(1, 0, 3, 2)

    return H


def newton(G, H):
    """Perform contraction between the inverse hessian and gradient
    to define search direction. Maybe add a preconditioner later."""

    N = G.shape[0]
    G = G.reshape(N * N)
    H = H.reshape(N * N, N * N)
    m = len(G)

    x, info = ssl.minres(H, -G)

    if info == 0:
        return x.reshape(N, N)
    else:
        raise Exception("Minres alg did not converge")


def cMF_opt(
    i1s,
    eri,
    D,
    C,
    Cmap,
    S,
    Norb,
    nclus,
    nstat,
    M,
    enr,
    ecore,
    anti,
    part,
    Sz,
    nstep_state=32,
    thrsh_state=1.0e-6,
    nstep_orb=256,
    thrsh_orb=1.0e-8,
    gnorm_orb=1.0e-6,
):
    """Perform a full cMF optimization with alternating steps for states and orbitals"""

    "Find initial states"
    if VERBOSE > 1:
        print("Calculating Initial cMF state")
    E0, D, orben = cMF_state(
        i1s,
        eri,
        D,
        Cmap,
        Norb,
        nclus,
        nstat,
        anti,
        nstep=nstep_state,
        thrsh=thrsh_state,
    )

    if nstep_orb == 0:
        "No orb opt"
        if VERBOSE > 1:
            print("No Orbital Optimization")
        if VERBOSE:
            print("Final cMF Energy: ", E0)
        return E0, D, C, orben, i1s, eri

    if VERBOSE > 1:
        print("Entering Orbital Optimization(Newton)")

    alpha = 2.0
    if VERBOSE > 1:
        print("Init Energy: ", E0 + enr + ecore)
    for n in range(nstep_orb):
        if VERBOSE > 1:
            print("Step ", n + 1)

        "Get orb grad and scale"
        if VERBOSE > 1:
            print("Calculating Gradient")
        G = cMF_orb_grad(i1s, eri, D, Cmap, nclus, Norb, anti)
        if VERBOSE > 4:
            print(G)

        "Calculate gnorm"
        Gnorm = 0.0
        for x in G:
            Gnorm += np.sum(np.sum(x**2))
        Gnorm = np.sqrt(Gnorm)

        if True:
            "Full newton"
            H = cMF_orb_hess(i1s, eri, Cmap, D, Norb, nclus, nstat, anti)
            G = newton(G, H)
            G *= alpha
            # TODO Implement good line search
        else:
            "Scale gradient for steepest descent"
            G *= -alpha
        if VERBOSE > 3:
            print("Current Step Direction:")
            print(G)
        if VERBOSE > 4:
            print("Current Hessian:")
            print(H)

        # Update orbs and ints
        C_ = update_orbs(C, G)
        i1s, eri = update_ints(C, C_, i1s, eri, S, M, nclus, Norb)
        if VERBOSE > 4:
            print("One electron Hamiltonian:")
            print(i1s)
            print("Two Electron Integrals:")
            print(eri)
        C = C_.copy()
        if VERBOSE > 3:
            print("Current Orbs:")
            print(C)

        "Get new cMF state"
        Energy, D, orben = cMF_state(
            i1s,
            eri,
            D,
            Cmap,
            Norb,
            nclus,
            nstat,
            anti,
            nstep=nstep_state,
            thrsh=thrsh_state,
        )
        if VERBOSE:
            print(Energy, Gnorm)
        if VERBOSE > 2:
            print("Current states:")
            print(D)
            print("Current Orbital Energies:")
            print(orben)
        if VERBOSE == 2:
            print("Current state:")
            if part[0] % 2:
                tmp1 = part[0] / 2
                tmp2 = int(tmp1 - Sz[0] / 2)
                tmp1 = int(tmp1 + Sz[0] / 2)
            else:
                tmp1 = part[0] // 2 + Sz[0]
                tmp2 = part[0] // 2 - Sz[0]
            tmp = int(ss.binom(Norb // 2, tmp1) * ss.binom(Norb // 2, tmp2))
            print(D[:, :tmp, 0].T)
            print("Current Orbital Energies:")
            print(orben[:, 0])

        # Check convergence
        if (np.abs(E0 - Energy) < thrsh_orb) and (Gnorm < gnorm_orb):
            status = True
            break
        elif n == nstep_orb - 1:
            print("Failed to converge orbitals")
            raise
        E0 = Energy

    if VERBOSE:
        print("Final cMF Energy: ", Energy)

    if status == True:
        return Energy, D, C, orben, i1s, eri


def reorder_states(D, orben, part, Sz, nstat, nclus):
    """Reorder the cluster states in each cluster to match
    the part and Sz vectors. For states with the same number of
    particles and Sz value, states will be order by energy.
    This function assumes that all states preserve these numbers
    and states are already ordered by energy."""
    for w in range(nclus):
        used = []
        for i in range(len(part)):
            p = part[i]
            z = Sz[i]
            match = []
            for j in range(nstat):
                if j in used:
                    continue
                a = np.where(np.abs(D[w, :, j]) > 1.0e-6)[0][0]
                p_ = part[a]
                z_ = Sz[a]
                if p == p_ and z == z_:
                    used.append(j)
                    break
                if j == nstat - 1:
                    raise Exception("Missing state in reorder")

        D[w] = D[w][:, used]
        orben[w] = orben[w, used]
    return D, orben


# fmt: off
def build_V(eri,i1s,nclus,nstat,Cmap,cluster,nleft,parity,batch_size,batch_index,normal=True,full_ham=False):
    """Build batches of integrals for the perturbation operator or Hamiltonian
    eri - Two electron integrals ordered (clus1,clus2,clus3,clus4,stat1,stat2,stat3,stat4)
    i1s - One electron integrals ordered (clus1,clus2,stat1,stat2)
    nclus - number of clusters
    nstat - number of states(assumed the same for all clusters)
    Cmap - mapping for annihilation operator in cMF state basis
    cluster - list of clusters on both sides(require that clusters on each side are in order)
    nleft - number of cluster indices on the bra
    parity - parity of states
    batch_size - number of cluster states gathered for each index at once(must evenly divide nstat)
    batch_index - which set of integrals are gathered at a time
    normal - False turns off the normal ordering, True returns normal ordered operator
    full_ham - True builds H, False builds V

    returns operator in the shape of batch_size
    """
    cluster = np.array(cluster)
    batch_size = np.array(batch_size)
    batch_index = np.array(batch_index)
    V = np.zeros(batch_size)

    "bra clusters"
    cleft = cluster[:nleft]
    "ket clusters"
    cright = cluster[nleft:]
    nright = len(cluster) - nleft

    "clusters in the ground state, not in cluster"
    extra = []
    for i in range(nclus):
        if not i in cluster:
            extra.append(i)

    if len(cright) != len(set(cright)) or len(cleft) != len(set(cleft)):
        raise Exception("Repeated cluster index in integral")
    for i in range(len(cleft)-1):
        if cleft[i+1] < cleft[i]:
            raise Exception("Cluster indices were given out of order")
    for i in range(len(cright)-1):
        if cright[i+1] < cright[i]:
            raise Exception("Cluster indices were given out of order")

    "clusters that are excited or deexcited from bra to ket"
    change = []
    for x in cleft:
        if not x in cright:
            change.append(x)
    for x in cright:
        if not x in cleft:
            change.append(x)

    diff = len(change)
    if diff > 4:
        "No interaction"
        return V

    for i in range(len(cluster)):
        if batch_index[i] >= (nstat) // batch_size[i]:
            raise Exception("batch index exceeds number of batches")

    done_normal = False

    if (not full_ham and not normal) and diff == 0:
        "Calculate <V> to remove normal ordering from V"
        V0 = 0.
        for p in range(nclus):
            for q in range(p+1,nclus):
                temp = np.einsum("ip,ir->pr",Cmap[:,0,:,p],Cmap[:,0,:,p],optimize=True)
                temp2 = np.einsum("pr,pqrs->qs",temp,eri[p,q,p,q],optimize=True)
                temp = np.einsum("kq,ks->qs",Cmap[:,0,:,q],Cmap[:,0,:,q],optimize=True)
                V0 -= np.einsum("qs,qs->",temp,temp2,optimize=True)

    if full_ham and normal and diff == 0:
        "Get <H> to normal order hamiltonian"
        H0 = 0.
        for p in range(nclus):
            temp = np.einsum("ip,pqrs->iqrs",Cmap[:,0,:,p],eri[p,p,p,p],optimize=True)
            temp = np.einsum("iqrs,jiq->jrs",temp,Cmap[:,:,:,p],optimize=True)
            temp = np.einsum("jrs,jks->kr",temp,Cmap[:,:,:,p],optimize=True)
            H0 += 0.25*np.einsum("kr,kr->",temp,Cmap[:,0,:,p],optimize=True)

            temp = np.einsum("ip,pq->iq",Cmap[:,0,:,p],i1s[p,p],optimize=True)
            H0 += np.einsum("iq,iq->",temp,Cmap[:,0,:,p],optimize=True)
            for q in range(p+1,nclus):
                temp = np.einsum("ip,ir->pr",Cmap[:,0,:,p],Cmap[:,0,:,p],optimize=True)
                temp2 = np.einsum("pr,pqrs->qs",temp,eri[p,q,p,q],optimize=True)
                temp = np.einsum("kq,ks->qs",Cmap[:,0,:,q],Cmap[:,0,:,q],optimize=True)
                H0 += np.einsum("qs,qs->",temp,temp2,optimize=True)

    if full_ham and diff == 0:
        "Terms for clusters not listed"
        t1 = 0.
        for y in extra:
            temp = np.einsum("ip,pqrs->iqrs",Cmap[:,0,:,y],eri[y,y,y,y],optimize=True)
            temp2 = np.einsum("iqrs,jiq->jrs",temp,Cmap[:,:,:,y],optimize=True)
            temp = np.einsum("jrs,jks->kr",temp2,Cmap[:,:,:,y],optimize=True)
            t1 += 0.25*np.einsum("kr,kr->",temp,Cmap[:,0,:,y],optimize=True)

            temp = np.einsum("ip,pq->iq",Cmap[:,0,:,y],i1s[y,y],optimize=True)
            t1 += np.einsum("iq,iq->",temp,Cmap[:,0,:,y],optimize=True)

            for z in extra:
                if z <= y:
                    continue
                temp = np.einsum("ip,ir->pr",Cmap[:,0,:,y],Cmap[:,0,:,y],optimize=True)
                temp2 = np.einsum("pr,pqrs->qs",temp,eri[y,z,y,z],optimize=True)
                temp = np.einsum("kq,ks->qs",Cmap[:,0,:,z],Cmap[:,0,:,z],optimize=True)
                t1 += np.einsum("qs,qs->",temp,temp2,optimize=True)

    for i,w in enumerate(set(cluster)):
        if diff > 0:
            if w != change[0]:
                continue
            if w in cleft:
                a = np.where(cleft == w)[0][0]
                bw1l = batch_size[a]*batch_index[a]
                bw2l = batch_size[a]*(batch_index[a]+1)
                bw1r = 0
                bw2r = 1
            else:
                a = np.where(cright == w)[0][0] + nleft
                bw1l = 0
                bw2l = 1
                bw1r = batch_size[a]*batch_index[a]
                bw2r = batch_size[a]*(batch_index[a]+1)
        else:
            a = np.where(cleft == w)[0][0]
            bw1l = batch_size[a]*batch_index[a]
            bw2l = batch_size[a]*(batch_index[a]+1)

            a = np.where(cright == w)[0][0] + nleft
            bw1r = batch_size[a]*batch_index[a]
            bw2r = batch_size[a]*(batch_index[a]+1)

        if full_ham and diff <= 1:
            V1 = np.zeros((bw2l-bw1l,bw2r-bw1r))

            "One cluster terms"
            temp = np.einsum("iap,pqrs->aiqrs",Cmap[:,bw1l:bw2l,:,w],eri[w,w,w,w],optimize=True)
            temp2 = np.einsum("aiqrs,jiq->ajrs",temp,Cmap[:,:,:,w],optimize=True)
            temp = np.einsum("ajrs,jks->akr",temp2,Cmap[:,:,:,w],optimize=True)
            V1 += 0.25*np.einsum("akr,kir->ai",temp,Cmap[:,bw1r:bw2r,:,w],optimize=True)

            temp = np.einsum("iap,pq->aiq",Cmap[:,bw1l:bw2l,:,w],i1s[w,w],optimize=True)
            V1 += np.einsum("aiq,ikq->ak",temp,Cmap[:,bw1r:bw2r,:,w],optimize=True)

            for y in extra:
                temp = np.einsum("iap,ijr->ajpr",Cmap[:,bw1l:bw2l,:,w],Cmap[:,bw1r:bw2r,:,w],optimize=True)
                temp2 = np.einsum("ajpr,pqrs->ajqs",temp,eri[w,y,w,y],optimize=True)
                temp = np.einsum("kq,ks->qs",Cmap[:,0,:,y],Cmap[:,0,:,y],optimize=True)
                V1 += np.einsum("qs,ajqs->aj",temp,temp2,optimize=True)

            if diff == 0 and i == 0:
                V1 += t1*np.eye(nstat)[bw1l:bw2l,bw1r:bw2r]

            if diff == 0 and normal and (not done_normal):
                V1 -= H0*np.eye(nstat)[bw1l:bw2l,bw1r:bw2r]
                done_normal = True

            "Bring in other clusters"
            l1 = ["b","c","d","e","f","g","h","t","u","v"]
            l2 = ["j","k","l","m","n","o","p","q","r","s"]
            s = "ai"
            o1 = ""
            o2 = ""
            op = [V1]
            q = 0
            temp = []
            for m,p in enumerate(cluster):
                if p == w:
                    if m < nleft:
                        o1 += "a"
                    else:
                        o2 += "i"
                    q += 1
                else:
                    if m < nleft:
                        a = np.where(cleft == p)[0][0]
                        bp1l = batch_size[a]*batch_index[a]
                        bp2l = batch_size[a]*(batch_index[a]+1)
                        a = np.where(cright == p)[0][0] + nleft
                        bp1r = batch_size[a]*batch_index[a]
                        bp2r = batch_size[a]*(batch_index[a]+1)
                        op.append(np.eye(nstat)[bp1l:bp2l,bp1r:bp2r])
                        s += "," + l1[m-q] + l2[m-q]
                        o1 += l1[m-q]
                        temp.append(l2[m-q])
                    else:
                        o2 += temp[0]
                        temp.pop(0)
            s += "->" + o1 + o2
            V += np.einsum(s,*op,optimize=True)

        for j,x in enumerate(set(cluster)):
            if diff > 1:
                if x != change[1]:
                    continue
                if x in cleft:
                    a = np.where(cleft == x)[0][0]
                    bx1l = batch_size[a]*batch_index[a]
                    bx2l = batch_size[a]*(batch_index[a]+1)
                    bx1r = 0
                    bx2r = 1
                else:
                    a = np.where(cright == x)[0][0] + nleft
                    bx1l = 0
                    bx2l = 1
                    bx1r = batch_size[a]*batch_index[a]
                    bx2r = batch_size[a]*(batch_index[a]+1)
            else:
                if x in change:
                    continue
                if diff == 0 and j <= i:
                    continue
                a = np.where(cleft == x)[0][0]
                bx1l = batch_size[a]*batch_index[a]
                bx2l = batch_size[a]*(batch_index[a]+1)

                a = np.where(cright == x)[0][0] + nleft
                bx1r = batch_size[a]*batch_index[a]
                bx2r = batch_size[a]*(batch_index[a]+1)

            if diff <= 2:
                "Two cluster terms"
                Ve = np.zeros((bw2l-bw1l,bx2l-bx1l,bw2r-bw1r,bx2r-bx1r))
                Vo = np.zeros((bw2l-bw1l,bx2l-bx1l,bw2r-bw1r,bx2r-bx1r))
                "+2|-2"
                temp = np.einsum("iap,jiq->ajpq",Cmap[:,bw1l:bw2l,:,w],Cmap[bw1r:bw2r,:,:,w],optimize=True)
                temp2 = np.einsum("ajpq,pqrs->ajrs",temp,eri[w,w,x,x],optimize=True)
                temp = np.einsum("bks,klr->blrs",Cmap[bx1l:bx2l,:,:,x],Cmap[:,bx1r:bx2r,:,x],optimize=True)
                Ve += 0.25*np.einsum("blrs,ajrs->abjl",temp,temp2,optimize=True)

                "-2|+2"
                temp = np.einsum("ibp,jiq->bjpq",Cmap[:,bx1l:bx2l,:,x],Cmap[bx1r:bx2r,:,:,x],optimize=True)
                temp2 = np.einsum("bjpq,pqrs->bjrs",temp,eri[x,x,w,w],optimize=True)
                temp = np.einsum("aks,klr->alrs",Cmap[bw1l:bw2l,:,:,w],Cmap[:,bw1r:bw2r,:,w],optimize=True)
                Ve += 0.25*np.einsum("alrs,bjrs->ablj",temp,temp2,optimize=True)

                "0|0"
                temp = np.einsum("iap,ijr->ajpr",Cmap[:,bw1l:bw2l,:,w],Cmap[:,bw1r:bw2r,:,w],optimize=True)
                temp2 = np.einsum("ajpr,pqrs->ajqs",temp,eri[w,x,w,x],optimize=True)
                temp = np.einsum("kbq,kls->blqs",Cmap[:,bx1l:bx2l,:,x],Cmap[:,bx1r:bx2r,:,x],optimize=True)
                Ve += np.einsum("blqs,ajqs->abjl",temp,temp2,optimize=True)

                if diff == 1 and not full_ham:
                    "Remove H0"
                    temp = np.einsum("iap,ijr->ajpr",Cmap[:,bw1l:bw2l,:,w],Cmap[:,bw1r:bw2r,:,w],optimize=True)
                    temp2 = np.einsum("ajpr,pqrs->ajqs",temp,eri[w,x,w,x],optimize=True)
                    temp = np.einsum("kq,ks->qs",Cmap[:,0,:,x],Cmap[:,0,:,x],optimize=True)
                    Ve -= np.einsum("qs,bl,ajqs->abjl",temp,np.eye(nstat)[bx1l:bx2l,bx1r:bx2r],temp2,optimize=True)

                if diff == 0 and not full_ham:
                    "Remove H0 and <V>"
                    temp = np.einsum("iap,ijr->ajpr",Cmap[:,bw1l:bw2l,:,w],Cmap[:,bw1r:bw2r,:,w],optimize=True)
                    temp2 = np.einsum("ajpr,pqrs->ajqs",temp,eri[w,x,w,x],optimize=True)
                    temp = np.einsum("kq,ks->qs",Cmap[:,0,:,x],Cmap[:,0,:,x],optimize=True)
                    Ve -= np.einsum("qs,bl,ajqs->abjl",temp,np.eye(nstat)[bx1l:bx2l,bx1r:bx2r],temp2,optimize=True)

                    temp = np.einsum("ip,ir,aj->ajpr",Cmap[:,0,:,w],Cmap[:,0,:,w],np.eye(nstat)[bw1l:bw2l,bw1r:bw2r],optimize=True)
                    temp2 = np.einsum("ajpr,pqrs->ajqs",temp,eri[w,x,w,x],optimize=True)
                    temp = np.einsum("kbq,kls->blqs",Cmap[:,bx1l:bx2l,:,x],Cmap[:,bx1r:bx2r,:,x],optimize=True)
                    Ve -= np.einsum("blqs,ajqs->abjl",temp,temp2,optimize=True)

                    temp = np.einsum("ip,ir->pr",Cmap[:,0,:,w],Cmap[:,0,:,w],optimize=True)
                    temp2 = np.einsum("pr,aj,pqrs->ajqs",temp,np.eye(nstat)[bw1l:bw2l,bw1r:bw2r],eri[w,x,w,x],optimize=True)
                    temp = np.einsum("kq,ks->qs",Cmap[:,0,:,x],Cmap[:,0,:,x],optimize=True)
                    Ve += np.einsum("qs,bl,ajqs->abjl",temp,np.eye(nstat)[bx1l:bx2l,bx1r:bx2r],temp2,optimize=True)

                    if (not normal) and (not done_normal):
                        "Put <V> back"
                        Ve += V0*np.einsum("aj,bl->abjl",np.eye(nstat)[bw1l:bw2l,bw1r:bw2r],np.eye(nstat)[bx1l:bx2l,bx1r:bx2r],optimize=True)
                        done_normal = True

                "+1|-1"
                temp = np.einsum("kap,pq->qak",Cmap[bw1r:bw2r,bw1l:bw2l,:,w],i1s[w,x],optimize=True)
                Vo += np.einsum("qak,blq->abkl",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)

                temp = np.einsum("kap,jkq->ajpq",Cmap[:,bw1l:bw2l,:,w],Cmap[:,:,:,w],optimize=True)
                temp2 = np.einsum("ajpq,jks->pqsak",temp,Cmap[:,bw1r:bw2r,:,w],optimize=True)
                temp = np.einsum("pqsak,pqrs->rak",temp2,eri[w,w,x,w],optimize=True)
                Vo += 0.5*np.einsum("rak,blr->abkl",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)

                temp = np.einsum("kbq,kjs->bjqs",Cmap[:,bx1l:bx2l,:,x],Cmap[:,:,:,x],optimize=True)
                temp2 = np.einsum("bjqs,jkr->qsrbk",temp,Cmap[:,bx1r:bx2r,:,x],optimize=True)
                temp = np.einsum("qsrbk,pqrs->pbk",temp2,eri[w,x,x,x],optimize=True)
                Vo += 0.5*np.einsum("pbk,lap->ablk",temp,Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)

                "-1|+1"
                temp = np.einsum("kbp,pq->qbk",Cmap[bx1r:bx2r,bx1l:bx2l,:,x],i1s[x,w],optimize=True)
                Vo -= np.einsum("qbk,alq->ablk",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)

                temp = np.einsum("kbp,jkq->bjpq",Cmap[:,bx1l:bx2l,:,x],Cmap[:,:,:,x],optimize=True)
                temp2 = np.einsum("bjpq,jks->pqsbk",temp,Cmap[:,bx1r:bx2r,:,x],optimize=True)
                temp = np.einsum("pqsbk,pqrs->rbk",temp2,eri[x,x,w,x],optimize=True)
                Vo -= 0.5*np.einsum("rbk,alr->ablk",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)

                temp = np.einsum("kaq,kjs->ajqs",Cmap[:,bw1l:bw2l,:,w],Cmap[:,:,:,w],optimize=True)
                temp2 = np.einsum("ajqs,jkr->qsrak",temp,Cmap[:,bw1r:bw2r,:,w],optimize=True)
                temp = np.einsum("qsrak,pqrs->pak",temp2,eri[x,w,w,w],optimize=True)
                Vo -= 0.5*np.einsum("pak,lbp->abkl",temp,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)

                "Add three cluster terms using extra"
                for y in extra:
                    temp = np.einsum("kq,ks->qs",Cmap[:,0,:,y],Cmap[:,0,:,y],optimize=True)
                    temp2 = np.einsum("qs,pqrs->pr",temp,eri[w,y,x,y],optimize=True)
                    temp = np.einsum("pr,iap->rai",temp2,Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)
                    Vo += np.einsum("rai,bkr->abik",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)

                    temp = np.einsum("kq,ks->qs",Cmap[:,0,:,y],Cmap[:,0,:,y],optimize=True)
                    temp2 = np.einsum("qs,pqrs->pr",temp,eri[x,y,w,y],optimize=True)
                    temp = np.einsum("pr,ibp->rbi",temp2,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)
                    Vo -= np.einsum("rbi,akr->abki",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)
                "Adjust for parity"
                if w < x:
                        Vo = np.einsum("abkl,ki->abil",Vo,np.diag(parity[w][bw1r:bw2r]),optimize=True)
                elif w > x:
                        Vo = np.einsum("abkl,bi->aikl",Vo,np.diag(parity[x][bx1l:bx2l]),optimize=True)

                "Bring in other clusters"
                l1 = ["c","d","e","f","g","h","t","u","v"]
                l2 = ["k","l","m","n","o","p","q","r","s"]
                s = "abij"
                o1 = ""
                o2 = ""
                ope = [Ve]
                opo = [Vo]
                q = 0
                temp = []
                for m,p in enumerate(cluster):
                    if p == w:
                        if m < nleft:
                            o1 += "a"
                        else:
                            o2 += "i"
                        q += 1
                    elif p == x:
                        if m < nleft:
                            o1 += "b"
                        else:
                            o2 += "j"
                        q += 1
                    else:
                        if m < nleft:
                            a = np.where(cleft == p)[0][0]
                            bp1l = batch_size[a]*batch_index[a]
                            bp2l = batch_size[a]*(batch_index[a]+1)
                            a = np.where(cright == p)[0][0] + nleft
                            bp1r = batch_size[a]*batch_index[a]
                            bp2r = batch_size[a]*(batch_index[a]+1)
                            ope.append(np.eye(nstat)[bp1l:bp2l,bp1r:bp2r])
                            if p == np.sort([w,x,p])[1]:
                                opo.append(np.diag(parity[p])[bp1l:bp2l,bp1r:bp2r])
                            else:
                                opo.append(np.eye(nstat)[bp1l:bp2l,bp1r:bp2r])
                            s += "," + l1[m-q] + l2[m-q]
                            o1 += l1[m-q]
                            temp.append(l2[m-q])
                        else:
                            o2 += temp[0]
                            temp.pop(0)
                s += "->" + o1 + o2
                V += np.einsum(s,*ope,optimize=True)
                V += np.einsum(s,*opo,optimize=True)

            for k,y in enumerate(set(cluster)):
                if diff > 2:
                    if y != change[2]:
                        continue
                    if y in cleft:
                        a = np.where(cleft == y)[0][0]
                        by1l = batch_size[a]*batch_index[a]
                        by2l = batch_size[a]*(batch_index[a]+1)
                        by1r = 0
                        by2r = 1
                    else:
                        a = np.where(cright == y)[0][0] + nleft
                        by1l = 0
                        by2l = 1
                        by1r = batch_size[a]*batch_index[a]
                        by2r = batch_size[a]*(batch_index[a]+1)
                else:
                    if y in change:
                        continue
                    if diff < 2 and k <= j:
                        continue
                    a = np.where(cleft == y)[0][0]
                    by1l = batch_size[a]*batch_index[a]
                    by2l = batch_size[a]*(batch_index[a]+1)

                    a = np.where(cright == y)[0][0] + nleft
                    by1r = batch_size[a]*batch_index[a]
                    by2r = batch_size[a]*(batch_index[a]+1)

                if diff <= 3:
                    "Three cluster terms"
                    V110 = np.zeros((bw2l-bw1l,bx2l-bx1l,by2l-by1l,bw2r-bw1r,bx2r-bx1r,by2r-by1r))
                    V101 = np.zeros((bw2l-bw1l,bx2l-bx1l,by2l-by1l,bw2r-bw1r,bx2r-bx1r,by2r-by1r))
                    V011 = np.zeros((bw2l-bw1l,bx2l-bx1l,by2l-by1l,bw2r-bw1r,bx2r-bx1r,by2r-by1r))

                    "+1|-1|0"
                    temp = np.einsum("icq,iks->ckqs",Cmap[:,by1l:by2l,:,y],Cmap[:,by1r:by2r,:,y],optimize=True)
                    temp = np.einsum("ckqs,pqrs->prck",temp,eri[w,y,x,y],optimize=True)
                    temp = np.einsum("prck,iap->raick",temp,Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)
                    V110 += np.einsum("raick,bjr->abcijk",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)

                    temp = np.einsum("icq,iks->ckqs",Cmap[:,by1l:by2l,:,y],Cmap[:,by1r:by2r,:,y],optimize=True)
                    temp = np.einsum("ckqs,pqrs->prck",temp,eri[x,y,w,y],optimize=True)
                    temp = np.einsum("prck,jbp->rbjck",temp,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)
                    V110 -= np.einsum("rbjck,air->abcijk",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)

                    temp = np.einsum("ibq,ijs->bjqs",Cmap[:,bx1l:bx2l,:,x],Cmap[:,bx1r:bx2r,:,x],optimize=True)
                    temp = np.einsum("bjqs,pqrs->prbj",temp,eri[w,x,y,x],optimize=True)
                    temp = np.einsum("prbj,iap->raibj",temp,Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)
                    V101 += np.einsum("raibj,ckr->abcijk",temp,Cmap[by1l:by2l,by1r:by2r,:,y],optimize=True)

                    temp = np.einsum("ibq,ijs->bjqs",Cmap[:,bx1l:bx2l,:,x],Cmap[:,bx1r:bx2r,:,x],optimize=True)
                    temp = np.einsum("bjqs,pqrs->prbj",temp,eri[y,x,w,x],optimize=True)
                    temp = np.einsum("prbj,kcp->rbjck",temp,Cmap[by1r:by2r,by1l:by2l,:,y],optimize=True)
                    V101 -= np.einsum("rbjck,air->abcijk",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)

                    temp = np.einsum("kaq,kis->aiqs",Cmap[:,bw1l:bw2l,:,w],Cmap[:,bw1r:bw2r,:,w],optimize=True)
                    temp = np.einsum("aiqs,pqrs->prai",temp,eri[x,w,y,w],optimize=True)
                    temp = np.einsum("prai,jbp->raibj",temp,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)
                    V011 += np.einsum("raibj,ckr->abcijk",temp,Cmap[by1l:by2l,by1r:by2r,:,y],optimize=True)

                    temp = np.einsum("kaq,kis->aiqs",Cmap[:,bw1l:bw2l,:,w],Cmap[:,bw1r:bw2r,:,w],optimize=True)
                    temp = np.einsum("aiqs,pqrs->prai",temp,eri[y,w,x,w],optimize=True)
                    temp = np.einsum("prai,kcp->raick",temp,Cmap[by1r:by2r,by1l:by2l,:,y],optimize=True)
                    V011 -= np.einsum("raick,bjr->abcijk",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)

                    "+2|-1|-1"
                    temp = np.einsum("kap,ikq->aipq",Cmap[:,bw1l:bw2l,:,w],Cmap[bw1r:bw2r,:,:,w],optimize=True)
                    temp = np.einsum("aipq,pqrs->srai",temp,eri[w,w,y,x],optimize=True)
                    temp = np.einsum("srai,bjs->rabij",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)
                    V011 += 0.5*np.einsum("rabij,ckr->abcijk",temp,Cmap[by1l:by2l,by1r:by2r,:,y],optimize=True)

                    temp = np.einsum("kbp,jkq->bjpq",Cmap[:,bx1l:bx2l,:,x],Cmap[bx1r:bx2r,:,:,x],optimize=True)
                    temp = np.einsum("bjpq,pqrs->srbj",temp,eri[x,x,y,w],optimize=True)
                    temp = np.einsum("srbj,ais->rabij",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)
                    V101 += 0.5*np.einsum("rabij,ckr->abcijk",temp,Cmap[by1l:by2l,by1r:by2r,:,y],optimize=True)

                    temp = np.einsum("icp,kiq->ckpq",Cmap[:,by1l:by2l,:,y],Cmap[by1r:by2r,:,:,y],optimize=True)
                    temp = np.einsum("ckpq,pqrs->srck",temp,eri[y,y,x,w],optimize=True)
                    temp = np.einsum("srck,ais->racik",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)
                    V110 += 0.5*np.einsum("racik,bjr->abcijk",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)

                    "+1|+1|-2"
                    temp = np.einsum("cis,ikr->srck",Cmap[by1l:by2l,:,:,y],Cmap[:,by1r:by2r,:,y],optimize=True)
                    temp = np.einsum("srck,pqrs->pqck",temp,eri[w,x,y,y],optimize=True)
                    temp = np.einsum("pqck,jbq->pbcjk",temp,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)
                    V110 += 0.5*np.einsum("pbcjk,iap->abcijk",temp,Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)

                    temp = np.einsum("bis,ijr->srbj",Cmap[bx1l:bx2l,:,:,x],Cmap[:,bx1r:bx2r,:,x],optimize=True)
                    temp = np.einsum("srbj,pqrs->pqbj",temp,eri[w,y,x,x],optimize=True)
                    temp = np.einsum("pqbj,kcq->pcbjk",temp,Cmap[by1r:by2r,by1l:by2l,:,y],optimize=True)
                    V101 += 0.5*np.einsum("pcbjk,iap->abcijk",temp,Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)

                    temp = np.einsum("ajs,jir->srai",Cmap[bw1l:bw2l,:,:,w],Cmap[:,bw1r:bw2r,:,w],optimize=True)
                    temp = np.einsum("srai,pqrs->pqai",temp,eri[x,y,w,w],optimize=True)
                    temp = np.einsum("pqai,kcq->pacik",temp,Cmap[by1r:by2r,by1l:by2l,:,y],optimize=True)
                    V011 += 0.5*np.einsum("pacik,jbp->abcijk",temp,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)

                    "Adjust for parity"
                    if w < x:
                        V110 = np.einsum("abcijk,ip->abcpjk",V110,np.diag(parity[w][bw1r:bw2r]),optimize=True)
                    else:
                        V110 = np.einsum("abcijk,bp->apcijk",V110,np.diag(parity[x][bx1l:bx2l]),optimize=True)
                    if y == np.sort([w,x,y])[1]:
                        V110 = np.einsum("abcijk,cp->abpijk",V110,np.diag(parity[y][by1l:by2l]),optimize=True)

                    if w < y:
                        V101 = np.einsum("abcijk,ip->abcpjk",V101,np.diag(parity[w][bw1r:bw2r]),optimize=True)
                    else:
                        V101 = np.einsum("abcijk,cp->abpijk",V101,np.diag(parity[y][by1l:by2l]),optimize=True)
                    if x == np.sort([w,x,y])[1]:
                        V101 = np.einsum("abcijk,bp->apcijk",V101,np.diag(parity[x][bx1l:bx2l]),optimize=True)

                    if x < y:
                        V011 = np.einsum("abcijk,jp->abcipk",V011,np.diag(parity[x][bx1r:bx2r]),optimize=True)
                    else:
                        V011 = np.einsum("abcijk,cp->abpijk",V011,np.diag(parity[y][by1l:by2l]),optimize=True)
                    if w == np.sort([w,x,y])[1]:
                        V011 = np.einsum("abcijk,ap->pbcijk",V011,np.diag(parity[w][bw1l:bw2l]),optimize=True)

                    "Bring in other clusters"
                    l1 = ["d","e","f","g","h","t","u","v"]
                    l2 = ["l","m","n","o","p","q","r","s"]
                    s = "abcijk"
                    o1 = ""
                    o2 = ""
                    op1 = [V011]
                    op2 = [V101]
                    op3 = [V110]

                    q = 0
                    temp = []
                    for m,p in enumerate(cluster):
                        if p == w:
                            if m < nleft:
                                o1 += "a"
                            else:
                                o2 += "i"
                            q += 1
                        elif p == x:
                            if m < nleft:
                                o1 += "b"
                            else:
                                o2 += "j"
                            q += 1
                        elif p == y:
                            if m < nleft:
                                o1 += "c"
                            else:
                                o2 += "k"
                            q += 1
                        else:
                            if m < nleft:
                                a = np.where(cleft == p)[0][0]
                                bp1l = batch_size[a]*batch_index[a]
                                bp2l = batch_size[a]*(batch_index[a]+1)
                                a = np.where(cright == p)[0][0] + nleft
                                bp1r = batch_size[a]*batch_index[a]
                                bp2r = batch_size[a]*(batch_index[a]+1)

                                if p == np.sort([w,x,p])[1]:
                                    op3.append(np.diag(parity[p])[bp1l:bp2l,bp1r:bp2r])
                                else:
                                    op3.append(np.eye(nstat)[bp1l:bp2l,bp1r:bp2r])

                                if p == np.sort([w,y,p])[1]:
                                    op2.append(np.diag(parity[p])[bp1l:bp2l,bp1r:bp2r])
                                else:
                                    op2.append(np.eye(nstat)[bp1l:bp2l,bp1r:bp2r])

                                if p == np.sort([y,x,p])[1]:
                                    op1.append(np.diag(parity[p])[bp1l:bp2l,bp1r:bp2r])
                                else:
                                    op1.append(np.eye(nstat)[bp1l:bp2l,bp1r:bp2r])

                                s += "," + l1[m-q] + l2[m-q]
                                o1 += l1[m-q]
                                temp.append(l2[m-q])
                            else:
                                o2 += temp[0]
                                temp.pop(0)
                    s += "->" + o1 + o2
                    V += np.einsum(s,*op1,optimize=True)
                    V += np.einsum(s,*op2,optimize=True)
                    V += np.einsum(s,*op3,optimize=True)

                for l,z in enumerate(set(cluster)):
                    if diff > 3:
                        if z != change[3]:
                            continue
                        if z in cleft:
                            a = np.where(cleft == z)[0][0]
                            bz1l = batch_size[a]*batch_index[a]
                            bz2l = batch_size[a]*(batch_index[a]+1)
                            bz1r = 0
                            bz2r = 1
                        else:
                            a = np.where(cright == z)[0][0] + nleft
                            bz1l = 0
                            bz2l = 1
                            bz1r = batch_size[a]*batch_index[a]
                            bz2r = batch_size[a]*(batch_index[a]+1)
                    else:
                        if z in change:
                            continue
                        if diff < 3 and l <= k:
                            continue
                        a = np.where(cleft == z)[0][0]
                        bz1l = batch_size[a]*batch_index[a]
                        bz2l = batch_size[a]*(batch_index[a]+1)

                        a = np.where(cright == z)[0][0] + nleft
                        bz1r = batch_size[a]*batch_index[a]
                        bz2r = batch_size[a]*(batch_index[a]+1)

                    "Four cluster terms"
                    V1111 = np.zeros((bw2l-bw1l,bx2l-bx1l,by2l-by1l,bz2l-bz1l,bw2r-bw1r,bx2r-bx1r,by2r-by1r,bz2r-bz1r))

                    "+1|+1|-1|-1"
                    temp = np.einsum("pqrs,iap->qrsai",eri[w,x,z,y],Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)
                    temp = np.einsum("qrsai,jbq->rsabij",temp,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)
                    temp = np.einsum("rsabij,cks->rabicjk",temp,Cmap[by1l:by2l,by1r:by2r,:,y],optimize=True)
                    V1111 += np.einsum("rabicjk,dlr->abcdijkl",temp,Cmap[bz1l:bz2l,bz1r:bz2r,:,z],optimize=True)

                    temp = np.einsum("pqrs,iap->qrsai",eri[w,y,z,x],Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)
                    temp = np.einsum("qrsai,kcq->rsaick",temp,Cmap[by1r:by2r,by1l:by2l,:,y],optimize=True)
                    temp = np.einsum("rsaick,bjs->rabcijk",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)
                    V1111 -= np.einsum("rabcijk,dlr->abcdijkl",temp,Cmap[bz1l:bz2l,bz1r:bz2r,:,z],optimize=True)

                    temp = np.einsum("pqrs,iap->qrsai",eri[w,z,y,x],Cmap[bw1r:bw2r,bw1l:bw2l,:,w],optimize=True)
                    temp = np.einsum("qrsai,ldq->rsaidl",temp,Cmap[bz1r:bz2r,bz1l:bz2l,:,z],optimize=True)
                    temp = np.einsum("rsaidl,bjs->rabdijl",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)
                    V1111 += np.einsum("rabdijl,ckr->abcdijkl",temp,Cmap[by1l:by2l,by1r:by2r,:,y],optimize=True)

                    temp = np.einsum("pqrs,kcp->qrsck",eri[y,x,z,w],Cmap[by1r:by2r,by1l:by2l,:,y],optimize=True)
                    temp = np.einsum("qrsck,jbq->rsbcjk",temp,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)
                    temp = np.einsum("rsbcjk,ais->rabcijk",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)
                    V1111 -= np.einsum("rabcijk,dlr->abcdijkl",temp,Cmap[bz1l:bz2l,bz1r:bz2r,:,z],optimize=True)

                    temp = np.einsum("pqrs,ldp->qrsdl",eri[z,x,y,w],Cmap[bz1r:bz2r,bz1l:bz2l,:,z],optimize=True)
                    temp = np.einsum("qrsdl,jbq->rsbdjl",temp,Cmap[bx1r:bx2r,bx1l:bx2l,:,x],optimize=True)
                    temp = np.einsum("rsbdjl,ais->rabdijl",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)
                    V1111 += np.einsum("rabdijl,ckr->abcdijkl",temp,Cmap[by1l:by2l,by1r:by2r,:,y],optimize=True)

                    temp = np.einsum("pqrs,kcp->qrsck",eri[y,z,x,w],Cmap[by1r:by2r,by1l:by2l,:,y],optimize=True)
                    temp = np.einsum("qrsck,ldq->rscdkl",temp,Cmap[bz1r:bz2r,bz1l:bz2l,:,z],optimize=True)
                    temp = np.einsum("rscdkl,ais->racdikl",temp,Cmap[bw1l:bw2l,bw1r:bw2r,:,w],optimize=True)
                    V1111 += np.einsum("racdikl,bjr->abcdijkl",temp,Cmap[bx1l:bx2l,bx1r:bx2r,:,x],optimize=True)

                    "Adjust for parity"
                    if w < x:
                        V1111 = np.einsum("abcdijkl,ip->abcdpjkl",V1111,np.diag(parity[w][bw1r:bw2r]),optimize=True)
                    else:
                        V1111 = np.einsum("abcdijkl,bp->apcdijkl",V1111,np.diag(parity[x][bx1l:bx2l]),optimize=True)
                    if w < y:
                        V1111 = np.einsum("abcdijkl,ip->abcdpjkl",V1111,np.diag(parity[w][bw1r:bw2r]),optimize=True)
                    else:
                        V1111 = np.einsum("abcdijkl,cp->abpdijkl",V1111,np.diag(parity[y][by1l:by2l]),optimize=True)
                    if w < z:
                        V1111 = np.einsum("abcdijkl,ip->abcdpjkl",V1111,np.diag(parity[w][bw1r:bw2r]),optimize=True)
                    else:
                        V1111 = np.einsum("abcdijkl,dp->abcpijkl",V1111,np.diag(parity[z][bz1l:bz2l]),optimize=True)
                    if x < y:
                        V1111 = np.einsum("abcdijkl,jp->abcdipkl",V1111,np.diag(parity[x][bx1r:bx2r]),optimize=True)
                    else:
                        V1111 = np.einsum("abcdijkl,cp->abpdijkl",V1111,np.diag(parity[y][by1l:by2l]),optimize=True)
                    if x < z:
                        V1111 = np.einsum("abcdijkl,jp->abcdipkl",V1111,np.diag(parity[x][bx1r:bx2r]),optimize=True)
                    else:
                        V1111 = np.einsum("abcdijkl,dp->abcpijkl",V1111,np.diag(parity[z][bz1l:bz2l]),optimize=True)
                    if y < z:
                        V1111 = np.einsum("abcdijkl,kp->abcdijpl",V1111,np.diag(parity[y][by1r:by2r]),optimize=True)
                    else:
                        V1111 = np.einsum("abcdijkl,dp->abcpijkl",V1111,np.diag(parity[z][bz1l:bz2l]),optimize=True)

                    "Bring in other clusters"
                    l1 = ["e","f","g","h","t","u","v"]
                    l2 = ["m","n","o","p","q","r","s"]
                    s = "abcdijkl"
                    o1 = ""
                    o2 = ""
                    op = [V1111]

                    q = 0
                    temp = []
                    for m,p in enumerate(cluster):
                        if p == w:
                            if m < nleft:
                                o1 += "a"
                            else:
                                o2 += "i"
                            q += 1
                        elif p == x:
                            if m < nleft:
                                o1 += "b"
                            else:
                                o2 += "j"
                            q += 1
                        elif p == y:
                            if m < nleft:
                                o1 += "c"
                            else:
                                o2 += "k"
                            q += 1
                        elif p == z:
                            if m < nleft:
                                o1 += "d"
                            else:
                                o2 += "l"
                            q += 1
                        else:
                            if m < nleft:
                                a = np.where(cleft == p)[0][0]
                                bp1l = batch_size[a]*batch_index[a]
                                bp2l = batch_size[a]*(batch_index[a]+1)
                                a = np.where(cright == p)[0][0] + nleft
                                bp1r = batch_size[a]*batch_index[a]
                                bp2r = batch_size[a]*(batch_index[a]+1)

                                if np.sum(np.array([w,x,y,z])>p) % 2 == 1:
                                    op.append(np.diag(parity[p])[bp1l:bp2l,bp1r:bp2r])
                                else:
                                    op.append(np.eye(nstat)[bp1l:bp2l,bp1r:bp2r])

                                s += "," + l1[m-q] + l2[m-q]
                                o1 += l1[m-q]
                                temp.append(l2[m-q])
                            else:
                                o2 += temp[0]
                                temp.pop(0)
                    s += "->" + o1 + o2
                    V += np.einsum(s,*op,optimize=True)
    return V


def PT(n,eri,i1s,Cmap,D,Norb,nclus,nstat,anti,parity,inter):
    """Compute up to the PTn correction"""
    if n > 3:
        raise Exception("Coded only up to PT3")

    H1c,H2c,den = build_h0(i1s,eri,D,Cmap,Norb,nclus,nstat,anti)
    Hc = np.einsum("wij,wik,wjl->wkl",H1c+H2c,D,D,optimize=True)
    Energy = MF_Energy(H1c,H2c,den,nclus)

    orben = np.zeros((nclus,nstat))
    for w in range(nclus):
        orben[w,:] = np.diag(Hc[w,:,:])

    print("PT0: ",np.sum(orben[:,0]))
    print("PT1: ",Energy-np.sum(orben[:,0]))
    if n == 1:
        return

    "Convert Cmap to cMF states"
    Cmap = np.einsum("ijpw,wik,wjl->klpw",Cmap,D,D,optimize=True)

    "Build energy denom"
    E1 = np.zeros((nclus,nstat-1))
    for w in range(nclus):
        E1[w,:] = Hc[w,0,0] - np.diag(Hc[w,1:,1:])
    E2 = np.add.outer(E1,E1).transpose(0,2,1,3)
    E3 = np.add.outer(E2,E1).transpose(0,1,4,2,3,5)
    E4 = np.add.outer(E3,E1).transpose(0,1,2,6,3,4,5,7)
    E1 = 1./E1
    E2 = 1./E2
    E3 = 1./E3
    E4 = 1./E4

    full = False
    norm = True

    "precompute end terms"
    V02 = np.zeros((nclus,nclus,nstat,nstat))
    V03 = np.zeros((nclus,nclus,nclus,nstat,nstat,nstat))
    V04 = np.zeros((nclus,nclus,nclus,nclus,nstat,nstat,nstat,nstat))
    for w in range(nclus):
        for x in range(w+1,nclus):
            V02[w,x] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x],0,parity,np.array([16,16]),np.zeros(2,dtype=int))
            for y in range(x+1,nclus):
                V03[w,x,y] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y],0,parity,np.array([16,16,16]),np.zeros(3,dtype=int))
                for z in range(y+1,nclus):
                    V04[w,x,y,z] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z],0,parity,np.array([16,16,16,16]),np.zeros(4,dtype=int))

    "Check PT2 value"
    En2 = np.einsum("wxab,wxab->",V02[:,:,1:,1:]**2,E2,optimize=True)
    En3 = np.einsum("wxyabc,wxyabc->",V03[:,:,:,1:,1:,1:]**2,E3,optimize=True)
    En4 = np.einsum("wxyzabcd,wxyzabcd->",V04[:,:,:,:,1:,1:,1:,1:]**2,E4,optimize=True)
    E1 = En2+En3+En4
    if inter:
        print("D ",En2)
        print("T ",En3)
        print("Q ",En4)
    print("PT2: ",E1)
    if n == 2:
        return

    "Precontract end terms"
    V02 = np.einsum("wxab,wxab->wxab",V02[:,:,1:,1:],E2,optimize=True)
    V03 = np.einsum("wxyabc,wxyabc->wxyabc",V03[:,:,:,1:,1:,1:],E3,optimize=True)
    V04 = np.einsum("wxyzabcd,wxyzabcd->wxyzabcd",V04[:,:,:,:,1:,1:,1:,1:],E4,optimize=True)

    "Calculate V^3 Energy"
    E2 = 0.
    E3 = 0.
    E4 = 0.
    E5 = 0.
    E6 = 0.
    E7 = 0.
    for w in range(nclus):
        for x in range(w+1,nclus):
            for ww in range(nclus):
                for xx in range(ww+1,nclus):
                    "DD"
                    V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,ww,xx],2,parity,np.array([16,16,16,16]),np.zeros(4,dtype=int),normal=norm,full_ham=full)
                    E2 += np.einsum("ab,abcd,cd->",V02[w,x],V[1:,1:,1:,1:],V02[ww,xx],optimize=True)
                    for y in range(x+1,nclus):
                        "TD"
                        V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,ww,xx],3,parity,np.array([16,16,16,16,16]),np.zeros(5,dtype=int),normal=norm,full_ham=full)
                        E4 += 2*np.einsum("abc,abcde,de->",V03[w,x,y],V[1:,1:,1:,1:,1:],V02[ww,xx],optimize=True)
                        for z in range(y+1,nclus):
                            "QD"
                            V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx],4,parity,np.array([16,16,16,16,16,16]),np.zeros(6,dtype=int),normal=norm,full_ham=full)
                            E5 += 2*np.einsum("abcd,abcdef,ef->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:],V02[ww,xx],optimize=True)
                        for yy in range(xx+1,nclus):
                            "TT"
                            V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,ww,xx,yy],3,parity,np.array([16,16,16,16,16,16]),np.zeros(6,dtype=int),normal=norm,full_ham=full)
                            E3 += np.einsum("abc,abcdef,def->",V03[w,x,y],V[1:,1:,1:,1:,1:,1:],V03[ww,xx,yy],optimize=True)
                            for z in range(y+1,nclus):
                                "QT"
                                V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx,yy],4,parity,np.array([16,16,16,16,16,16,16]),np.zeros(7,dtype=int),normal=norm,full_ham=full)
                                E6 += 2*np.einsum("abcd,abcdefg,efg->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:,1:],V03[ww,xx,yy],optimize=True)
                                for zz in range(yy+1,nclus):
                                    "QQ"
                                    for i in range(1,16):
                                        tmp = np.zeros(8,dtype=int)
                                        tmp[0] = i
                                        V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx,yy,zz],4,parity,np.array([1,16,16,16,16,16,16,16]),tmp,normal=norm,full_ham=full)
                                        E7 += np.einsum("abcd,abcdefgh,efgh->",V04[w,x,y,z,i-1:i],V[0:1,1:,1:,1:,1:,1:,1:,1:],V04[ww,xx,yy,zz],optimize=True)

    if inter:
        print("DD ",E2)
        print("DT ",E4)
        print("DQ ",E5)
        print("TT ",E3)
        print("TQ ",E6)
        print("QQ ",E7)
    print("PT3: ",E2+E3+E4+E5+E6+E7)
    if n == 3:
        return


def driver():
    a=int(input("molecule ::\t"))
    for x in range(-15,-5):
        molecule = x
        if x != -a :
            continue

        molecule = x
        n=None
        if False:
            "temporary because the cluster is packed"
            import sys
            molecule = int(sys.argv[1])
            n = int(sys.argv[2])

        "Unrestricted not implemented in this code, only use with Carlos' code"
        unrestrict = False

        "Write integrals for Carlos' code"
        write=False

        "Reorder states after optimization by particle and Sz. False leaves them in order of orbital energy"
        order = True

        "use anti-symmetrized integrals"
        anti = True

        "Convergence parameters for optimization"
        nstep_state=128
        nstep_orb=64
        thrsh_state=1e-10
        thrsh_orb=1e-10
        gnorm_orb=1.e-10

        "Calculate PT up to order do_PT(do_PT=0 does no PT calculation)"
        do_PT = 2

        "Print cPT intermediates"
        inter = False


        #Orbs per cluster(assume half filled, Sz=0 mean field ,hard coded for Norb = 4)
        Norb = 4
        assert Norb == 4

        if VERBOSE:
            print(f"Molecule: {Molecules[molecule]}")
            print("Control Settings:")
            print(f"VERBOSE = {VERBOSE}")
            print(f"molecule = {molecule}")
            print(f"unrestrict = {unrestrict}")
            print(f"Norb = {Norb}")
            print(f"order = {order}")
            print(f"anti = {anti}")
            print(f"nstep_state={nstep_state}")
            print(f"nstep_orb={nstep_orb}")
            print(f"thrsh_state={thrsh_state}")
            print(f"thrsh_orb={thrsh_orb}")
            print(f"gnorm_orb={gnorm_orb}")
            print(f"do_PT = {do_PT}")


        "Get cluster orbitals"
        if VERBOSE > 1:
            print("Preparing Orbitals")
        cf, U, cf0, S, ic0, N, M, mol, mf = prepare_cf(molecule,unrestrict,n=n,FCI=False)
        
        print(cf) #Printing orbital here
        
        nclus = M*2//Norb
        
        nstat = 2**(Norb)
        if VERBOSE:
            print(f"Number of Clusters: {nclus}")
            print(f"Number of States per Cluster: {nstat}")

        "Get integrals"
        if VERBOSE > 1:
            print("Calculating Integrals")
        i1s, eri, enr, ecore = get_ints(molecule,mol,mf,cf,cf0,ic0)
        # print(ecore)
        # print(enr)
        # raise
        
        "write integrals"
        if write:
            if VERBOSE > 1:
                print("Writing Integrals")
            write_ints(i1s,eri,U,enr,ecore)

        "Expand integrals"
        if VERBOSE > 1:
            print("Expanding Integrals")
        cf, i1s, eri = expand_ints(cf,i1s,eri,M)

        "Get some things for cMF"
        Cmap, part, Sz, D, i1s, eri = prepare_cMF(i1s,eri,Norb,nclus,nstat,anti)


        if VERBOSE:
            print("Starting cMF Optimization")

        Energy,D,C,orben,i1s,eri = cMF_opt(i1s,eri,D,cf,Cmap,S,Norb,nclus,nstat,M,enr,ecore,anti,part,Sz, \
                nstep_state=nstep_state,thrsh_state=thrsh_state,nstep_orb=nstep_orb,thrsh_orb=thrsh_orb,gnorm_orb=gnorm_orb)

        if order:
            D,orben = reorder_states(D,orben,part,Sz,nstat,nclus)


        if VERBOSE == 1:
            print("Final state:")
            if part[0] % 2:
                tmp1 = part[0]/2
                tmp2 = int(tmp1 - Sz[0]/2)
                tmp1 = int(tmp1 + Sz[0]/2)
            else:
                tmp1 = part[0]//2 + Sz[0]
                tmp2 = part[0]//2 - Sz[0]
            tmp = int(ss.binom(Norb//2,tmp1)*ss.binom(Norb//2,tmp2))
            print(D[:,:tmp,0].T)
            print("Orbital Energies:")
            print(orben[:,0])

        if VERBOSE > 1:
            print("Final states:")
            print(D)
            print("Orbital Energies:")
            print(orben)

        print("Final Energy: ",Energy+enr+ecore)

        if do_PT:
            if VERBOSE > 1:
                print("Making parity vector")

            "Make parity vector"
            parity = np.ones((nclus,nstat))
            part_D = np.zeros((nclus,nstat))
            Sz_D = np.zeros((nclus,nstat))
            for i in range(nclus):
                for j in range(nstat):
                    a = D[i,:,j]
                    a = np.where(np.abs(a) >1.e-6)[0][0]
                    part_D[i,j] = part[a]
                    Sz_D[i,j] = Sz[a]
                    if part[a] % 2:
                        parity[i,j] *= -1
            if VERBOSE > 2:
                print("Parity vector:")
                print(parity)

            if VERBOSE:
                print(f"Calculating up to PT{do_PT} energy")

            PT(do_PT,eri,i1s,Cmap,D,Norb,nclus,nstat,anti,parity,inter)

    return Energy,D,C,orben,i1s,eri,nclus,Norb,nstat,Cmap,part,Sz,anti

if __name__ == "__main__":
    Energy,D,C,orben,i1s,eri,nclus,Norb,nstat,Cmap,part,Sz,anti = driver()

if False:
    "Old Code"
    #print(D)
    #print(orben)
    raise

    #print(np.einsum("ij,ki,kl,lp->jp",Cmap[:,:,0,0],Cmap[:,:,3,0],Cmap[:,:,1,0],Cmap[:,:,2,0]))
    #print(np.einsum("ij,ki,kl,lp->jp",Cmap[:,:,2,0],Cmap[:,:,1,0],Cmap[:,:,1,0],Cmap[:,:,2,0]))
    #print(np.einsum("ij,ki,kl,lp->jp",Cmap[:,:,3,0],Cmap[:,:,0,0],Cmap[:,:,0,0],Cmap[:,:,3,0]))
    #raise
    #D[1][:,1] = D[1][:,1]*-1
    print(D[0])
    #print(orben[0])
    print(part)
    print(Sz)
    #raise

    def PT2(eri,i1s,Cmap,D):
        """Compute the PT2 correction"""

        H1c,H2c,den = build_h0(i1s,eri,D)
        Hc = np.einsum("wij,wik,wjl->wkl",H1c+H2c,D,D,optimize=True)

        "Convert Cmap"
        Cmap = np.einsum("ijpw,wik,wjl->klpw",Cmap,D,D,optimize=True)

        "Build energy denom"
        E1 = np.zeros((nclus,nstat-1))
        for w in range(nclus):
            E1[w,:] = Hc[w,0,0] - np.diag(Hc[w,1:,1:])
        E2 = np.add.outer(E1,E1).transpose(0,2,1,3)
        E3 = np.add.outer(E2,E1).transpose(0,1,4,2,3,5)
        E4 = np.add.outer(E3,E1).transpose(0,1,2,6,3,4,5,7)
        E1 = 1./E1
        E2 = 1./E2
        E3 = 1./E3
        E4 = 1./E4
        for w in range(nclus):
            E2[w,w] *= 0

        "2 clus term"
        Energy = 0.
        for w in range(nclus):
            for x in range(w):
                "+2|-2"
                temp = np.einsum("ip,jiq->jpq",Cmap[:,0,:,w],Cmap[:,:,:,w],optimize=True)
                temp2 = np.einsum("jpq,pqrs->jrs",temp,eri[w,w,x,x],optimize=True)
                temp = np.einsum("ks,klr->lrs",Cmap[0,:,:,x],Cmap[:,:,:,x],optimize=True)
                temp3 = np.einsum("lrs,jrs->jl",temp,temp2,optimize=True)
                Energy += 2*0.0625*np.einsum("jl,jl->",temp3[1:,1:]**2,E2[w,x],optimize=True)

                "0|0"
                temp = np.einsum("ip,ijr->jpr",Cmap[:,0,:,w],Cmap[:,:,:,w],optimize=True)
                temp2 = np.einsum("jpr,pqrs->jqs",temp,eri[w,x,w,x],optimize=True)
                temp = np.einsum("kq,kls->lqs",Cmap[:,0,:,x],Cmap[:,:,:,x],optimize=True)
                temp3 = np.einsum("lqs,jqs->jl",temp,temp2,optimize=True)
                Energy += 2*0.5*np.einsum("jl,jl->",temp3[1:,1:]**2,E2[w,x],optimize=True)

                "+1|-1"
                V = np.zeros((nstat,nstat))
                temp = np.einsum("kp,pq->qk",Cmap[:,0,:,w],i1s[w,x],optimize=True)
                V += np.power(-1,x<w)*np.einsum("qk,lq->kl",temp,Cmap[0,:,:,x],optimize=True)

                temp = np.einsum("kp,jkq->jpq",Cmap[:,0,:,w],Cmap[:,:,:,w],optimize=True)
                temp2 = np.einsum("jpq,jks->pqsk",temp,Cmap[:,:,:,w],optimize=True)
                temp = np.einsum("pqsk,pqrs->rk",temp2,eri[w,w,x,w],optimize=True)
                V += np.power(-1,x<w)*0.5*np.einsum("rk,lr->kl",temp,Cmap[0,:,:,x],optimize=True)

                temp = np.einsum("kq,kjs->jqs",Cmap[:,0,:,x],Cmap[:,:,:,x],optimize=True)
                temp2 = np.einsum("jqs,jkr->qsrk",temp,Cmap[:,:,:,x],optimize=True)
                temp = np.einsum("qsrk,pqrs->pk",temp2,eri[w,x,x,x],optimize=True)
                V += np.power(-1,x<w)*0.5*np.einsum("pk,lp->lk",temp,Cmap[:,0,:,w],optimize=True)

                Energy += 2*0.5*np.einsum("jl,jl->",V[1:,1:]**2,E2[w,x],optimize=True)

                "-1|+1"
                V = np.zeros((nstat,nstat))
                temp = np.einsum("kp,pq->qk",Cmap[:,0,:,x],i1s[x,w],optimize=True)
                V += np.power(-1,x>w)*np.einsum("qk,lq->lk",temp,Cmap[0,:,:,w],optimize=True)

                temp = np.einsum("kq,kjs->jqs",Cmap[:,0,:,w],Cmap[:,:,:,w],optimize=True)
                temp2 = np.einsum("jqs,jkr->qsrk",temp,Cmap[:,:,:,w],optimize=True)
                temp = np.einsum("qsrk,pqrs->pk",temp2,eri[x,w,w,w],optimize=True)
                V += np.power(-1,x>w)*0.5*np.einsum("pk,lp->kl",temp,Cmap[:,0,:,x],optimize=True)

                temp = np.einsum("kp,jkq->jpq",Cmap[:,0,:,x],Cmap[:,:,:,x],optimize=True)
                temp2 = np.einsum("jpq,jks->pqsk",temp,Cmap[:,:,:,x],optimize=True)
                temp = np.einsum("pqsk,pqrs->rk",temp2,eri[x,x,w,x],optimize=True)
                V += np.power(-1,x>w)*0.5*np.einsum("rk,lr->lk",temp,Cmap[0,:,:,w],optimize=True)

                Energy += 2*0.5*np.einsum("jl,jl->",V[1:,1:]**2,E2[w,x],optimize=True)

                for y in range(x):
                    "+1|-1|0"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kq,kjs->jqs",Cmap[:,0,:,y],Cmap[:,:,:,y],optimize=True)
                    temp2 = np.einsum("jqs,pqrs->prj",temp,eri[w,y,x,y],optimize=True)
                    temp = np.einsum("prj,ip->rij",temp2,Cmap[:,0,:,w],optimize=True)
                    V += np.einsum("rij,kr->ikj",temp,Cmap[0,:,:,x],optimize=True)
                    Energy += np.einsum("ikj,ikj->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "-1|+1|0"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kq,kjs->jqs",Cmap[:,0,:,y],Cmap[:,:,:,y],optimize=True)
                    temp2 = np.einsum("jqs,pqrs->prj",temp,eri[x,y,w,y],optimize=True)
                    temp = np.einsum("prj,ip->rij",temp2,Cmap[:,0,:,x],optimize=True)
                    V -= np.einsum("rij,kr->kij",temp,Cmap[0,:,:,w],optimize=True)
                    Energy += np.einsum("ikj,ikj->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "-1|0|+1"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kq,kjs->jqs",Cmap[:,0,:,x],Cmap[:,:,:,x],optimize=True)
                    temp2 = np.einsum("jqs,pqrs->prj",temp,eri[y,x,w,x],optimize=True)
                    temp = np.einsum("prj,kp->rjk",temp2,Cmap[:,0,:,y],optimize=True)
                    V -= np.einsum("rjk,ir->ijk",temp,Cmap[0,:,:,w],optimize=True)
                    Energy += np.einsum("ijk,ijk->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "+1|0|-1"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kq,kjs->jqs",Cmap[:,0,:,x],Cmap[:,:,:,x],optimize=True)
                    temp2 = np.einsum("jqs,pqrs->prj",temp,eri[w,x,y,x],optimize=True)
                    temp = np.einsum("prj,kp->rjk",temp2,Cmap[:,0,:,w],optimize=True)
                    V += np.einsum("rjk,ir->kji",temp,Cmap[0,:,:,y],optimize=True)
                    Energy += np.einsum("ijk,ijk->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "0|-1|+1"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kq,kjs->jqs",Cmap[:,0,:,w],Cmap[:,:,:,w],optimize=True)
                    temp2 = np.einsum("jqs,pqrs->prj",temp,eri[y,w,x,w],optimize=True)
                    temp = np.einsum("prj,kp->rjk",temp2,Cmap[:,0,:,y],optimize=True)
                    V -= np.einsum("rjk,ir->jik",temp,Cmap[0,:,:,x],optimize=True)
                    Energy += np.einsum("ijk,ijk->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "0|+1|-1"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kq,kjs->jqs",Cmap[:,0,:,w],Cmap[:,:,:,w],optimize=True)
                    temp2 = np.einsum("jqs,pqrs->prj",temp,eri[x,w,y,w],optimize=True)
                    temp = np.einsum("prj,kp->rjk",temp2,Cmap[:,0,:,x],optimize=True)
                    V += np.einsum("rjk,ir->jki",temp,Cmap[0,:,:,y],optimize=True)
                    Energy += np.einsum("ijk,ijk->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "+2|-1|-1"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kp,ikq->ipq",Cmap[:,0,:,w],Cmap[:,:,:,w],optimize=True)
                    temp2 = np.einsum("ipq,pqrs->sri",temp,eri[w,w,x,y],optimize=True)
                    temp = np.einsum("sri,js->rij",temp2,Cmap[0,:,:,y],optimize=True)
                    V -= 0.5*np.einsum("rij,kr->ikj",temp,Cmap[0,:,:,x],optimize=True)
                    Energy += np.einsum("ijk,ijk->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "-1|+2|-1"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kp,ikq->ipq",Cmap[:,0,:,x],Cmap[:,:,:,x],optimize=True)
                    temp2 = np.einsum("ipq,pqrs->sri",temp,eri[x,x,y,w],optimize=True)
                    temp = np.einsum("sri,js->rij",temp2,Cmap[0,:,:,w],optimize=True)
                    V += 0.5*np.einsum("rij,kr->jik",temp,Cmap[0,:,:,y],optimize=True)
                    Energy += np.einsum("jik,jik->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "-1|-1|+2"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("kp,ikq->ipq",Cmap[:,0,:,y],Cmap[:,:,:,y],optimize=True)
                    temp2 = np.einsum("ipq,pqrs->sri",temp,eri[y,y,x,w],optimize=True)
                    temp = np.einsum("sri,js->rij",temp2,Cmap[0,:,:,w],optimize=True)
                    V += 0.5*np.einsum("rij,kr->jki",temp,Cmap[0,:,:,x],optimize=True)
                    Energy += np.einsum("jik,jik->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "+1|+1|-2"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("is,ikr->srk",Cmap[0,:,:,y],Cmap[:,:,:,y],optimize=True)
                    temp2 = np.einsum("srk,pqrs->pqk",temp,eri[w,x,y,y],optimize=True)
                    temp = np.einsum("pqk,jq->pjk",temp2,Cmap[:,0,:,x],optimize=True)
                    V += 0.5*np.einsum("pjk,ip->ijk",temp,Cmap[:,0,:,w],optimize=True)
                    Energy += np.einsum("ijk,ijk->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "+1|-2|+1"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("is,ikr->srk",Cmap[0,:,:,x],Cmap[:,:,:,x],optimize=True)
                    temp2 = np.einsum("srk,pqrs->pqk",temp,eri[w,y,x,x],optimize=True)
                    temp = np.einsum("pqk,jq->pjk",temp2,Cmap[:,0,:,y],optimize=True)
                    V += 0.5*np.einsum("pjk,ip->ikj",temp,Cmap[:,0,:,w],optimize=True)
                    Energy += np.einsum("ijk,ijk->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    "-2|+1|+1"
                    V = np.zeros((nstat,nstat,nstat))
                    temp = np.einsum("is,ikr->srk",Cmap[0,:,:,w],Cmap[:,:,:,w],optimize=True)
                    temp2 = np.einsum("srk,pqrs->pqk",temp,eri[x,y,w,w],optimize=True)
                    temp = np.einsum("pqk,jq->pjk",temp2,Cmap[:,0,:,y],optimize=True)
                    V += 0.5*np.einsum("pjk,ip->kij",temp,Cmap[:,0,:,x],optimize=True)
                    Energy += np.einsum("ijk,ijk->",V[1:,1:,1:]**2,E3[w,x,y],optimize=True)

                    for z in range(y):
                        "+1|+1|-1|-1"
                        V = np.zeros((nstat,nstat,nstat,nstat))
                        temp = np.einsum("pqrs,ip->qrsi",eri[w,x,z,y],Cmap[:,0,:,w],optimize=True)
                        temp2 = np.einsum("qrsi,jq->rsij",temp,Cmap[:,0,:,x],optimize=True)
                        temp = np.einsum("rsij,ks->rijk",temp2,Cmap[0,:,:,y],optimize=True)
                        V += np.einsum("rijk,lr->ijkl",temp,Cmap[0,:,:,z],optimize=True)
                        Energy += np.einsum("ijkl,ijkl",V[1:,1:,1:,1:]**2,E4[w,x,y,z],optimize=True)

                        "+1|-1|+1|-1"
                        V = np.zeros((nstat,nstat,nstat,nstat))
                        temp = np.einsum("pqrs,ip->qrsi",eri[w,y,z,x],Cmap[:,0,:,w],optimize=True)
                        temp2 = np.einsum("qrsi,jq->rsij",temp,Cmap[:,0,:,y],optimize=True)
                        temp = np.einsum("rsij,ks->rijk",temp2,Cmap[0,:,:,x],optimize=True)
                        V -= np.einsum("rijk,lr->ikjl",temp,Cmap[0,:,:,z],optimize=True)
                        Energy += np.einsum("ijkl,ijkl",V[1:,1:,1:,1:]**2,E4[w,x,y,z],optimize=True)

                        "+1|-1|-1|+1"
                        V = np.zeros((nstat,nstat,nstat,nstat))
                        temp = np.einsum("pqrs,ip->qrsi",eri[w,z,x,y],Cmap[:,0,:,w],optimize=True)
                        temp2 = np.einsum("qrsi,jq->rsij",temp,Cmap[:,0,:,z],optimize=True)
                        temp = np.einsum("rsij,ks->rijk",temp2,Cmap[0,:,:,y],optimize=True)
                        V -= np.einsum("rijk,lr->ilkj",temp,Cmap[0,:,:,x],optimize=True)
                        Energy += np.einsum("ijkl,ijkl",V[1:,1:,1:,1:]**2,E4[w,x,y,z],optimize=True)

                        "-1|+1|+1|-1"
                        V = np.zeros((nstat,nstat,nstat,nstat))
                        temp = np.einsum("pqrs,ip->qrsi",eri[y,x,z,w],Cmap[:,0,:,y],optimize=True)
                        temp2 = np.einsum("qrsi,jq->rsij",temp,Cmap[:,0,:,x],optimize=True)
                        temp = np.einsum("rsij,ks->rijk",temp2,Cmap[0,:,:,w],optimize=True)
                        V += np.einsum("rijk,lr->kjil",temp,Cmap[0,:,:,z],optimize=True)
                        Energy += np.einsum("ijkl,ijkl",V[1:,1:,1:,1:]**2,E4[w,x,y,z],optimize=True)

                        "-1|+1|-1|+1"
                        V = np.zeros((nstat,nstat,nstat,nstat))
                        temp = np.einsum("pqrs,ip->qrsi",eri[z,x,w,y],Cmap[:,0,:,z],optimize=True)
                        temp2 = np.einsum("qrsi,jq->rsij",temp,Cmap[:,0,:,x],optimize=True)
                        temp = np.einsum("rsij,ks->rijk",temp2,Cmap[0,:,:,y],optimize=True)
                        V -= np.einsum("rijk,lr->ljki",temp,Cmap[0,:,:,w],optimize=True)
                        Energy += np.einsum("ijkl,ijkl",V[1:,1:,1:,1:]**2,E4[w,x,y,z],optimize=True)

                        "-1|-1|+1|+1"
                        V = np.zeros((nstat,nstat,nstat,nstat))
                        temp = np.einsum("pqrs,ip->qrsi",eri[y,z,x,w],Cmap[:,0,:,y],optimize=True)
                        temp2 = np.einsum("qrsi,jq->rsij",temp,Cmap[:,0,:,z],optimize=True)
                        temp = np.einsum("rsij,ks->rijk",temp2,Cmap[0,:,:,w],optimize=True)
                        V += np.einsum("rijk,lr->klij",temp,Cmap[0,:,:,x],optimize=True)
                        Energy += np.einsum("ijkl,ijkl",V[1:,1:,1:,1:]**2,E4[w,x,y,z],optimize=True)

        print(Energy)
        return

    def PT3(eri,i1s,Cmap,D,Norb,nclus,nstat,anti):
        """Compute the PT3 correction"""

        H1c,H2c,den = build_h0(i1s,eri,D,Cmap,Norb,nclus,nstat,anti)
        Hc = np.einsum("wij,wik,wjl->wkl",H1c+H2c,D,D,optimize=True)

        "Convert Cmap"
        Cmap = np.einsum("ijpw,wik,wjl->klpw",Cmap,D,D,optimize=True)

        "Build energy denom"
        E1 = np.zeros((nclus,nstat-1))
        for w in range(nclus):
            E1[w,:] = Hc[w,0,0] - np.diag(Hc[w,1:,1:])
        E2 = np.add.outer(E1,E1).transpose(0,2,1,3)
        E3 = np.add.outer(E2,E1).transpose(0,1,4,2,3,5)
        E4 = np.add.outer(E3,E1).transpose(0,1,2,6,3,4,5,7)
        E1 = 1./E1
        E2 = 1./E2
        E3 = 1./E3
        E4 = 1./E4
        for w in range(nclus):
            E2[w,w] *= 0

        Energy = 0.

        "precompute end terms"
        V02 = np.zeros((nclus,nclus,nstat,nstat))
        V03 = np.zeros((nclus,nclus,nclus,nstat,nstat,nstat))
        V04 = np.zeros((nclus,nclus,nclus,nclus,nstat,nstat,nstat,nstat))
        for w in range(nclus):
            for x in range(w+1,nclus):
                V02[w,x] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x],0,parity,np.array([16,16]),np.zeros(2,dtype=int))
                for y in range(x+1,nclus):
                    V03[w,x,y] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y],0,parity,np.array([16,16,16]),np.zeros(3,dtype=int))
                    for z in range(y+1,nclus):
                        V04[w,x,y,z] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z],0,parity,np.array([16,16,16,16]),np.zeros(4,dtype=int))

        V03 *= 0
        V04 *= 0
        "Check PT2 value"
        En2 = np.einsum("wxab,wxab->",V02[:,:,1:,1:]**2,E2,optimize=True)
        En3 = np.einsum("wxyabc,wxyabc->",V03[:,:,:,1:,1:,1:]**2,E3,optimize=True)
        En4 = np.einsum("wxyzabcd,wxyzabcd->",V04[:,:,:,:,1:,1:,1:,1:]**2,E4,optimize=True)
        print(En2)
        print(En3)
        print("PT2 Energy: ",Energy+En2+En3+En4)

        "Calculate PT3 Energy"
        V02 = np.einsum("wxab,wxab->wxab",V02[:,:,1:,1:],E2,optimize=True)
        V03 = np.einsum("wxyabc,wxyabc->wxyabc",V03[:,:,:,1:,1:,1:],E3,optimize=True)
        V04 = np.einsum("wxyzabcd,wxyzabcd->wxyzabcd",V04[:,:,:,:,1:,1:,1:,1:],E4,optimize=True)

        Energy = 0.
        for w in range(nclus):
            for x in range(w+1,nclus):
                for ww in range(nclus):
                    for xx in range(ww+1,nclus):
                        "DD"
                        V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,ww,xx],2,parity,np.array([16,16,16,16]),np.zeros(4,dtype=int))
                        Energy += np.einsum("ab,abcd,cd->",V02[w,x],V[1:,1:,1:,1:],V02[ww,xx],optimize=True)
                        continue
                        for y in range(x+1,nclus):
                            "TD"
                            V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,ww,xx],3,parity,np.array([16,16,16,16,16]),np.zeros(5,dtype=int))
                            Energy += 2*np.einsum("abc,abcde,de->",V03[w,x,y],V[1:,1:,1:,1:,1:],V02[ww,xx],optimize=True)
                            for z in range(y+1,nclus):
                                "QD"
                                V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx],4,parity,np.array([16,16,16,16,16,16]),np.zeros(6,dtype=int))
                                Energy += 2*np.einsum("abcd,abcdef,de->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:],V02[ww,xx],optimize=True)
                            for yy in range(xx+1,nclus):
                                "TT"
                                V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,ww,xx,yy],3,parity,np.array([16,16,16,16,16,16]),np.zeros(6,dtype=int))
                                Energy += np.einsum("abc,abcdef,def->",V03[w,x,y],V[1:,1:,1:,1:,1:,1:],V03[ww,xx,yy],optimize=True)
                                for z in range(y+1,nclus):
                                    "QT"
                                    V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx,yy],4,parity,np.array([16,16,16,16,16,16,16]),np.zeros(7,dtype=int))
                                    Energy += 2*np.einsum("abcd,abcdefg,efg->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:,1:],V03[ww,xx,yy],optimize=True)
                                    for zz in range(yy+1,nclus):
                                        "QQ"
                                        V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx,yy,zz],4,parity,np.array([16,16,16,16,16,16,16,16]),np.zeros(8,dtype=int))
                                        Energy += np.einsum("abcd,abcdefgh,efgh->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:,1:,1:],V04[ww,xx,yy,zz],optimize=True)
        print("PT3 Energy: ",Energy)
        raise
        return

    def Hn(eri,i1s,Cmap,D,Norb,nclus,nstat,anti,parity):
        """Compute <H^n>"""

        H1c,H2c,den = build_h0(i1s,eri,D,Cmap,Norb,nclus,nstat,anti)
        Hc = np.einsum("wij,wik,wjl->wkl",H1c+H2c,D,D,optimize=True)
        Energy = MF_Energy(H1c,H2c,den,nclus)
        print("cMF",Energy)
        Energy = Energy**2
        print(Energy)

        orben = np.zeros((nclus,nstat))
        for w in range(nclus):
            orben[w,:] = np.diag(Hc[w,:,:])

    #    temp = np.einsum("icp,kiq->ckpq",Cmap[:,:,:,2],Cmap[:,:,:,2],optimize=True)
    #    print(temp)
    #    print(temp[0,6])
    #    print(np.where(temp))
    #    raise
        "Convert Cmap"
        Cmap = np.einsum("ijpw,wik,wjl->klpw",Cmap,D,D,optimize=True)

        two = False
        full = False
        norm = True
        V_only = False
        W_only = False
        three = False
        four = False

    #    eri*=0

        "precompute end terms"
        V02 = np.zeros((nclus,nclus,nstat,nstat))
        V03 = np.zeros((nclus,nclus,nclus,nstat,nstat,nstat))
        V04 = np.zeros((nclus,nclus,nclus,nclus,nstat,nstat,nstat,nstat))
        for w in range(nclus):
            for x in range(w+1,nclus):
                V02[w,x] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x],0,parity,np.array([16,16]),np.zeros(2,dtype=int))
                for y in range(x+1,nclus):
                    V03[w,x,y] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y],0,parity,np.array([16,16,16]),np.zeros(3,dtype=int))
                    for z in range(y+1,nclus):
                        V04[w,x,y,z] = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z],0,parity,np.array([16,16,16,16]),np.zeros(4,dtype=int))
    #    print(V03[0,1,2,:,:,1])
    #    raise

    #    V03 *= 0
    #    V04 *= 0
        "Check PT2 value"
        En2 = np.einsum("wxab->",V02[:,:,1:,1:]**2,optimize=True)
        En3 = np.einsum("wxyabc->",V03[:,:,:,1:,1:,1:]**2,optimize=True)
        En4 = np.einsum("wxyzabcd->",V04[:,:,:,:,1:,1:,1:,1:]**2,optimize=True)
        print(En2)
        print(En3)
        print(En4)
        print("PT2 Energy: ",En2+En3+En4)
        print("Total Energy: ",Energy+En2+En3+En4)
        E1 = En2+En3+En4
        V02 = V02[:,:,1:,1:]
        V03 = V03[:,:,:,1:,1:,1:]
        V04 = V04[:,:,:,:,1:,1:,1:,1:]

        V_only = False
        W_only = False
        three = False
        two = False
        four = False
        "Calculate H^3 Energy"
        E = MF_Energy(H1c,H2c,den,nclus)
        Energy = E**3 + 2*E*E1
        E2 = 0.
        E3 = 0.
        E4 = 0.
        E5 = 0.
        E6 = 0.
        E7 = 0.
        E2_ = 0.
        E3_ = 0.
        E4_ = 0.
        temp1 = 0.
        temp2 = 0.
        temp3 = 0.
        temp12 = 0.
        temp13 = 0.
        temp23 = 0.
        for w in range(nclus):
            for x in range(w+1,nclus):
                for ww in range(nclus):
                    for xx in range(ww+1,nclus):
                        "DD"
                        V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,ww,xx],2,parity,np.array([16,16,16,16]),np.zeros(4,dtype=int),normal=norm,full_ham=full)
                        E2 += np.einsum("ab,abcd,cd->",V02[w,x],V[1:,1:,1:,1:],V02[ww,xx],optimize=True)
                        for y in range(x+1,nclus):
                            "TD"
                            V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,ww,xx],3,parity,np.array([16,16,16,16,16]),np.zeros(5,dtype=int),normal=norm,full_ham=full)
                            E4 += 2*np.einsum("abc,abcde,de->",V03[w,x,y],V[1:,1:,1:,1:,1:],V02[ww,xx],optimize=True)
                            for z in range(y+1,nclus):
                                "QD"
                                V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx],4,parity,np.array([16,16,16,16,16,16]),np.zeros(6,dtype=int),normal=norm,full_ham=full)
    #                            Energy += 2*np.einsum("abcd,abcdef,de->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:],V02[ww,xx],optimize=True)
                                E5 += 2*np.einsum("abcd,abcdef,ef->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:],V02[ww,xx],optimize=True)
                            for yy in range(xx+1,nclus):
                                "TT"
                                V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,ww,xx,yy],3,parity,np.array([16,16,16,16,16,16]),np.zeros(6,dtype=int),normal=norm,full_ham=full)
                                E3 += np.einsum("abc,abcdef,def->",V03[w,x,y],V[1:,1:,1:,1:,1:,1:],V03[ww,xx,yy],optimize=True)
                                for z in range(y+1,nclus):
                                    "QT"
                                    V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx,yy],4,parity,np.array([16,16,16,16,16,16,16]),np.zeros(7,dtype=int),normal=norm,full_ham=full)
    #                                Energy += 2*np.einsum("abcd,abcdefg,efg->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:,1:],V03[ww,xx,yy],optimize=True)
                                    E6 += 2*np.einsum("abcd,abcdefg,efg->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:,1:],V03[ww,xx,yy],optimize=True)
                                    for zz in range(yy+1,nclus):
                                        "QQ"
                                        for i in range(1,16):
                                            tmp = np.zeros(8,dtype=int)
                                            tmp[0] = i
                                            V = build_V(eri,i1s,nclus,nstat,Cmap,[w,x,y,z,ww,xx,yy,zz],4,parity,np.array([1,16,16,16,16,16,16,16]),tmp,normal=norm,full_ham=full)
        #                                    Energy += np.einsum("abcd,abcdefgh,efgh->",V04[w,x,y,z],V[1:,1:,1:,1:,1:,1:,1:,1:],V04[ww,xx,yy,zz],optimize=True)
                                            E7 += np.einsum("abcd,abcdefgh,efgh->",V04[w,x,y,z,i-1:i],V[0:1,1:,1:,1:,1:,1:,1:,1:],V04[ww,xx,yy,zz],optimize=True)

        temp = 0.
        for w in range(nclus):
            temp -= np.trace((0.5*H2c[w]).dot(den[w]))
        print("<V>",temp)
        print("<V>",E-np.sum(orben[:,0]))
        print("VV", E1)
        print("DD", E2)
        print("TD12",temp12)
        print("TD13",temp13)
        print("TD23",temp23)
        print("DD",E2)
        print("DT",E4/2)
        print("TT",E3)
        print("DQ",E5/2)
        print("TQ",E6/2)
        print("QQ",E7)
        print("VVV",E2+E3+E4+E5+E6+E7)
        print("HH",E*E+E1)
        print("HHH",E*E*E+2*E*E1+E2+E3+E4+E5+E6+E7)
        raise
        print("<V>?",temp)
        print("<H>",E)
        print("<V>",E-np.sum(orben[:,0]))
        print("<H0>",np.sum(orben[:,0]))
        print("HH",E*E+E1)
        print("HV",E*E -E*(np.sum(orben[:,0])) +E1)
        print("VVtot",(E-np.sum(orben[:,0]))**2 + E1)
        print("VV", E1)
        print("HHH",E*E*E+2*E*E1+E2+E3+E4)
        print("VVV",E2+E3+E4)
        print("DD",E2)
        print("DT",E4/2)
        print("TT",E3)
        raise
        print("DT+DD",E2+E4/2)
        print("TT+2DT+DD",E2+E3+E4)
        print("TT+2DT",E3+E4)
        print(E2_+E3_+E4_-E2-E3-E4)
        raise

        print("DD",E2)
        renorm = (E - np.sum(orben[:,0]))*E1
        print("2HHH+renorm",E*E*E+2*E*E1+E2+renorm)
        print("HHH+renorm",E*E*E+2*E*E1+E2+renorm+E3+E4)
        print("VVV",E2+renorm+E3+E4)
        print("VVe",E1)
        print("trip",E3+En3)
        print("DD",E2)
        print(E4/2+E2)
        raise
        print("VVV",E2+renorm)
        print("Total+renorm",Energy+E2+renorm)
        raise
        print("<V>",E-np.sum(orben[:,0]))
        print("VRRV",E1)
        print("renorm",renorm)
        print("VVV",E2+renorm)
        Energy += E2
        print("PT3 Energy: ",Energy)
        raise

        return

    "Make parity vector"
    parity = np.ones((nclus,nstat))
    part_D = np.zeros((nclus,nstat))
    Sz_D = np.zeros((nclus,nstat))
    for i in range(nclus):
        for j in range(nstat):
            a = D[i,:,j]
            a = np.where(np.abs(a) >1.e-6)[0][0]
            part_D[i,j] = part[a]
            Sz_D[i,j] = Sz[a]
            if part[a] % 2:
                parity[i,j] *= -1
    #PT2(eri,i1s,Cmap,D)
    #PT3(eri,i1s,Cmap,D,Norb,nclus,nstat,anti)
    Hn(eri,i1s,Cmap,D,Norb,nclus,nstat,anti,parity)
    #PT(2,eri,i1s,Cmap,D,Norb,nclus,nstat,anti,parity)
    raise


    #Make two cluster terms for comparison
    #print(D[0])
    H1c,H2c,den = build_h0(i1s,eri,D)
    Hc = np.einsum("wij,wik,wjl->wkl",H1c+H2c,D,D,optimize=True)
    Cmap = np.einsum("ijpw,wik,wjl->klpw",Cmap,D,D,optimize=True)
    #print(np.diag(Hc[0]))
    qwd = np.ones((Norb,Norb)) - np.eye(Norb)
    V = np.zeros((nclus,nclus,nstat,nstat,nstat,nstat))
    "diag terms"
    temp = np.einsum("jipw,jkrw->ikprw",Cmap,Cmap,optimize=True)
    temp2 = np.einsum("jiqw,jksw->ikqsw",Cmap,Cmap,optimize=True)
    V += np.einsum("ikprw,jlqsx,wxwxpqrs->wxijkl",temp,temp2,eri,optimize=True)
    for w in range(nclus):
        for x in range(nclus):
            "+2|-2"
            temp = np.einsum("jip,kjq->pqik",Cmap[:,:,:,w],Cmap[:,:,:,w],optimize=True)
            temp2 = np.einsum("pqik,pqrs->rsik",temp,eri[w,w,x,x],optimize=True)
            temp = np.einsum("jks,klr->rsjl",Cmap[:,:,:,x],Cmap[:,:,:,x],optimize=True)
            V[w,x] += 0.25*np.einsum("rsjl,rsik->ijkl",temp,temp2,optimize=True)
            V[x,w] += 0.25*np.einsum("rsjl,rsik->jilk",temp,temp2,optimize=True)

            "0|0"
    #        temp = np.einsum("jip,jkr,pr->prik",Cmap[:,:,:,w],Cmap[:,:,:,w],qwd,optimize=True)
    #        temp2 = np.einsum("prik,pqrs->qsik",temp,eri[w,x,w,x],optimize=True)
    #        temp = np.einsum("kjq,kls,qs->qsjl",Cmap[:,:,:,x],Cmap[:,:,:,x],qwd,optimize=True)
    #        V[w,x] += np.einsum("qsjl,qsik->ijkl",temp,temp2,optimize=True)

            "+1|-1"
            temp = np.einsum("kip,pq->qik",Cmap[:,:,:,w],i1s[w,x],optimize=True)
            V[w,x] += np.power(-1,x<w)*np.einsum("qik,jlq->ijkl",temp,Cmap[:,:,:,x],optimize=True)

            temp = np.einsum("kip,jkq->jpqi",Cmap[:,:,:,w],Cmap[:,:,:,w],optimize=True)
            temp2 = np.einsum("jpqi,jks->pqsik",temp,Cmap[:,:,:,w],optimize=True)
            temp = np.einsum("pqsik,pqrs->rik",temp2,eri[w,w,x,w],optimize=True)
            V[w,x] += np.power(-1,x<w)*0.5*np.einsum("rik,jlr->ijkl",temp,Cmap[:,:,:,x],optimize=True)

            temp = np.einsum("kiq,kjs->jqsi",Cmap[:,:,:,x],Cmap[:,:,:,x],optimize=True)
            temp2 = np.einsum("jqsi,jkr->qsrik",temp,Cmap[:,:,:,x],optimize=True)
            temp = np.einsum("qsrik,pqrs->pik",temp2,eri[w,x,x,x],optimize=True)
            V[w,x] += np.power(-1,x<w)*0.5*np.einsum("pik,ljp->jilk",temp,Cmap[:,:,:,w],optimize=True)

            "-1|+1"
            temp = np.einsum("kip,pq->qik",Cmap[:,:,:,x],i1s[x,w],optimize=True)
            V[w,x] -= np.power(-1,x>w)*np.einsum("qik,jlq->jilk",temp,Cmap[:,:,:,w],optimize=True)

            temp = np.einsum("kiq,kjs->jqsi",Cmap[:,:,:,w],Cmap[:,:,:,w],optimize=True)
            temp2 = np.einsum("jqsi,jkr->qsrik",temp,Cmap[:,:,:,w],optimize=True)
            temp = np.einsum("qsrik,pqrs->pik",temp2,eri[x,w,w,w],optimize=True)
            V[w,x] -= np.power(-1,x>w)*0.5*np.einsum("pik,ljp->ijkl",temp,Cmap[:,:,:,x],optimize=True)

            temp = np.einsum("kip,jkq->jpqi",Cmap[:,:,:,x],Cmap[:,:,:,x],optimize=True)
            temp2 = np.einsum("jpqi,jks->pqsik",temp,Cmap[:,:,:,x],optimize=True)
            temp = np.einsum("pqsik,pqrs->rik",temp2,eri[x,x,w,x],optimize=True)
            V[w,x] -= np.power(-1,x>w)*0.5*np.einsum("rik,jlr->jilk",temp,Cmap[:,:,:,w],optimize=True)

    print(part)
    print(Sz)
    "reorder V for comparison"
    x = [2,2,2,2,1,1,1,1,3,3,0,3,3,2,2,4]
    y = [0,0,0,0,-1,-1,1,1,-1,-1,0,1,1,-2,2,0]
    order = []
    used = []
    for i in range(nstat):
        for j in range(nstat):
            if j in used:
                continue
            if part_D[0][j] == x[i] and Sz_D[0][j] == y[i]:
                order.append(j)
                used.append(j)
                continue

    print(order)
    V = V[:,:,order,:,:,:]
    V = V[:,:,:,order,:,:]
    V = V[:,:,:,:,order,:]
    V = V[:,:,:,:,:,order]
    print(orben[0][order])

    #print(V[0,1,0,1])
    V = build_V(eri,i1s,nclus,nstat,Cmap,[0,1,0,1],2,parity,[16,16,16,16],[0,0,0,0])
    #print(V[0,1,:,:])
    #V = build_V(eri,i1s,nclus,nstat,Cmap,[0,1],2,parity,[16,16],[0,0])
    V = V[order,:,:,:]
    V = V[:,order,:,:]
    V = V[:,:,order,:]
    V = V[:,:,:,order]
    #print(V)
    print(V[0,1])
    "Build energy denom"
    #E1 = np.zeros((nclus,nstat-1))
    #for w in range(nclus):
    #    E1[w,:] = Hc[w,0,0] - np.diag(Hc[w,1:,1:])
    #E2 = np.add.outer(E1,E1).transpose(0,2,1,3)
    #E1 = 1./E1
    #E2 = 1./E2
    #for w in range(nclus):
    #    E2[w,w] *= 0

    #print("----")
    #e2 = 0.5*np.einsum("wxab,wxab->",V[:,:,0,0,1:,1:]**2,E2,optimize=True)
    #print(e2)

    from scipy.io import FortranFile
    en = FortranFile("en.dat", "r").read_reals(dtype=float)
    orb_en = en.reshape((-1, nstat,), order="F",)
    matel2 = FortranFile("matel2.dat", "r").read_reals(dtype=float)
    matel2 = matel2.reshape((nclus, nstat, nstat,) * 2, order="F",)
    V = matel2.copy().transpose(1, 4, 2, 5, 0, 3)
    W = matel2.copy().transpose(1, 4, 2, 5, 0, 3)
    state = np.array([2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 4, 1, 1, 2, 2, 0])%2
    """Add signs to integrals"""
    # print(V[:,:,:,:,1,0])
    for x in range(nclus):
        for y in range(nclus):
            for a in range(16):
                for b in range(16):
                    if state[a] != state[b]:
                        V[a, :, b, :, x, y] *= 0
                    else:
                        W[a, :, b, :, x, y] *= 0
    V += V.transpose(1, 0, 3, 2, 5, 4)
    W -= W.transpose(1, 0, 3, 2, 5, 4)
    #V += W.copy()
    #Vx = V.copy().transpose(0, 1, 3, 2, 4, 5)
    #Wx = W.copy().transpose(0, 1, 3, 2, 4, 5)
    orb = orb_en[0]
    print(orb)
    E1 = np.zeros((nclus,nstat-1))
    for i in range(nclus):
        E1[i] = orb_en[i,0] - orb_en[i,1:]
    E2 = np.add.outer(E1,E1).transpose(0,2,1,3)
    E2 = 1./E2

    print(V[0,1,:,:,0,1])
    print(V[0,0,0,0,0,1])
    "Calculate PT3 Energy"
    Energy = 0.
    a = b = c = d = 0.
    for w in range(nclus):
        for x in range(w+1,nclus):
            "DD"
            a += np.einsum("ab,abcd,cd,ab,cd->",V[0,0,1:,1:,w,x],V[1:,1:,1:,1:,w,x],V[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            a -= np.einsum("ab,abcd,cd,ab,cd->",W[0,0,1:,1:,w,x],W[1:,1:,1:,1:,w,x],V[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            a -= np.einsum("ab,abcd,cd,ab,cd->",W[0,0,1:,1:,w,x],V[1:,1:,1:,1:,w,x],W[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            a -= np.einsum("ab,abcd,cd,ab,cd->",V[0,0,1:,1:,w,x],W[1:,1:,1:,1:,w,x],W[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)

            b += V[0,0,0,0,w,x]*np.einsum("ab,ab,ab,ab->",V[0,0,1:,1:,w,x],V[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            b -= W[0,0,0,0,w,x]*np.einsum("ab,ab,ab,ab->",W[0,0,1:,1:,w,x],V[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            b -= V[0,0,0,0,w,x]*np.einsum("ab,ab,ab,ab->",W[0,0,1:,1:,w,x],W[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            b -= W[0,0,0,0,w,x]*np.einsum("ab,ab,ab,ab->",V[0,0,1:,1:,w,x],W[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)

            c -= np.einsum("ab,bd,ad,ab,ad->",V[0,0,1:,1:,w,x],V[0,1:,0,1:,w,x],V[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            c += np.einsum("ab,bd,ad,ab,ad->",W[0,0,1:,1:,w,x],W[0,1:,0,1:,w,x],V[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            c += np.einsum("ab,bd,ad,ab,ad->",W[0,0,1:,1:,w,x],V[0,1:,0,1:,w,x],W[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            c += np.einsum("ab,bd,ad,ab,ad->",V[0,0,1:,1:,w,x],W[0,1:,0,1:,w,x],W[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)

            d -= np.einsum("ab,ac,cb,ab,cb->",V[0,0,1:,1:,w,x],V[1:,0,1:,0,w,x],V[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            d += np.einsum("ab,ac,cb,ab,cb->",W[0,0,1:,1:,w,x],W[1:,0,1:,0,w,x],V[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            d += np.einsum("ab,ac,cb,ab,cb->",W[0,0,1:,1:,w,x],V[1:,0,1:,0,w,x],W[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
            d += np.einsum("ab,ac,cb,ab,cb->",V[0,0,1:,1:,w,x],W[1:,0,1:,0,w,x],W[1:,1:,0,0,w,x],E2[w,x],E2[w,x],optimize=True)
    print(a)
    print(b)
    print(c)
    print(d)

    Energy = a + b + c + d
    print("PT3 Energy: ",Energy)

if False:
    "Old Code"
    param = [["diene_6g.mat","sto-3g",4,4],["diene_2z.mat","cc-pvdz",4,4],
             ["benzene_3g.mat","sto-3g",6,6],["benzene_2z.mat","cc-pvdz",6,6],
             ["triene_6g.mat","sto-3g",6,6],["triene_6g.mat","cc-pvdz",6,6],
             ["C12.mat","cc-pvdz",12,12],["C16.mat","cc-pvdz",16,16],["C20.mat","cc-pvdz",20,20],
             ["coro.mat","cc-pvdz",24,24],["coro.mat","cc-pvdz",24,24],
             ["circoro.mat","cc-pvdz",54,54],["tetraene.mat","cc-pvdz",8,8],
             ["pentaene.mat","cc-pvdz",10,10],["pyrene_geom.mat","cc-pvdz",16,16],["pyrene_geom.mat","cc-pvdz",16,16],
             ["C54.mat","cc-pvdz",54,54],["C44.mat","cc-pvdz",44,44],["C30.mat","cc-pvdz",30,30],["C24.mat","cc-pvdz",24,24],["ethyl.mat","cc-pvdz",2,2],
             ["tetraene_1z.mat","sto-3g",8,8]]
    orb_swap = [[],[[16,19]],[[16,18]],[[16,18],[23,29]],[],[[23,29]],[[45,53],[47,56],[48,57]],[[60,70],[61,73],[62,75],[64,76]],[],[],[],[],[[31,36],[32,39]],[[38,42],[39,46],[40,47]],[],[],[],[],[],[],[],[]]
    atom_list = [[0,3,5,7],[0,3,5,7],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,3,5,7,9,11],[0,3,5,7,9,11],
            [0,3,5,7,9,11,13,15,17,19,21,23],[0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39],
            [0,1,2,3,4,5,23,6,12,13,7,14,15,8,16,17,9,18,19,10,20,21,11,22],
            [3,9,10,4,18,19,5,11,22,23,6,0,1,7,14,15,8,2,17,16,13,12,21,20],
            [0,1,2,3,4,5,12,6,23,48,51,53,7,13,27,25,24,14,15,8,16,31,59,30,9,17,34,61,35,18,19,10,20,41,39,36,11,21,44,63,47,22,56,54,58,28,32,60,37,62,42,45,49,64],
            [0,3,5,7,9,11,13,15],[0,3,5,7,9,11,13,15,17,19],
            [0,1,6,10,11,7,12,13,8,14,15,9,4,5,2,3],[2,7,12,13,8,3,11,10,6,1,0,5,4,9,15,14],#[1,2,7,11,10,6,3,4,9,15,14,8,5,0,13,12],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59],
            [0,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47],[0,3],[0,3,5,7,9,11,13,15]]
    unrestrict = False
    molecule = 1
    if molecule > 0:
        "Important parameters"
        mel_file,basis,N,M = param[molecule-1]

        "Get info from matfile to build mol"
        mel = qcm.MatEl(file=mel_file)
        coord = mel.c.reshape((-1,3),order="C")
        atm_chg = mel.atmchg
        nocc = mel.ne//2
        no = M
        ic0 = (nocc - N//2)
        mol = gto.Mole()
        mol.atom = []
        for i in range(len(atm_chg)):
            mol.atom.append([int(atm_chg[i]),tuple(coord[i,:])])
    #    mol.unit = "angstrom"
        mol.unit = "au"
        mol.basis = basis
        mol.max_memory = 110000

        mol.build()
        S = mol.intor_symmetric("int1e_ovlp")
        nvir = mol.nao - nocc

        xbst = mol._bas[:,1]
        nctr = mol._bas[:,3]
        xbs = np.empty(0)
        atm = np.empty(0)
        for i in range(len(nctr)):
            for _ in range(nctr[i]):
                xbs = np.hstack((xbs,xbst[i]))
                atm = np.hstack((atm,mol._bas[i,0]))
        xbs = np.array([int(x) for x in xbs])
        atm = np.array([int(x) for x in atm])


        "Make ideal orbitals for x-y planar conugated systems"
        atoms = atom_list[molecule-1]
        if np.allclose(np.sort(atoms),np.arange(M)):
            carbons = atoms
        else:
            carbons = np.ones(M,dtype=int)*-1
            ind = -1
            used = []
            for i in range(M):
                val = M*100
                for j in range(M):
                    if (atoms[j] < val) and (not j in used):
                        ind = j
                        val = atoms[j]
                carbons[ind] = i
                used.append(ind)

        "Find 2pz orbitals"
        pz = []
        used = []
        x = 0
        for i in range(len(xbs)):
            if ((xbs[i] == 1) and (atm[i] in atoms) and (not atm[i] in used)):
                pz.append(x+2)
                used.append(atm[i])
            x += 2*xbs[i] + 1
        if len(pz) != M:
            raise

        if not molecule in [7,8,9,10,11,12,15,16,17,18,19,20]:
            "Get orbitals from CAS calculation"
            mf = scf.RHF(mol).run()
            C = mf.mo_coeff
            swap = orb_swap[molecule-1]
            for x in swap:
                print(x)
                C[:,x] = C[:,list(reversed(x))]
    #        print(ic0)
    #        print(C[pz,ic0:ic0+N])
    #        raise
            cas = mcscf.CASSCF(mf,N,M).run()
            print(cas.ci)
            C = cas.mo_coeff
            print("CAS Energy",cas.e_cas)
            cf0 = C

            "Make ideal orbitals"
            D = np.zeros((mol.nao,M))
            for i in range(M//2):
                D[pz[carbons[2*i]],2*i] = 1
                D[pz[carbons[2*i+1]],2*i] = 1
                D[pz[carbons[2*i]],2*i+1] = 1
                D[pz[carbons[2*i+1]],2*i+1] = -1
            S_ = D.T.dot(S).dot(D)
            val,vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
            D = D.dot(S_)

            "Project ideal orbitals onto active space"
            Cac = C[:,ic0:ic0+M]
            Dac  = Cac.dot(Cac.T).dot(S).dot(D)
            S_ = Dac.T.dot(S).dot(Dac)
            val,vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
            cf = Dac.dot(S_)
            U = np.eye(M)
            print(cf[pz,:])
        else:
            "Get orbitals from UHF orbitals"

            "Get UHF guess with broken symmetry"
            mf = scf.RHF(mol).run()
            C = mf.mo_coeff
            T = np.zeros((M),dtype=int)
            q = 0
            di = np.diag(C[pz,:].T.dot(S[pz,:][:,pz]).dot(C[pz,:]))
            for i in range(mol.nao):
                if di[i] > 0.1:
                    T[q] = i
                    q += 1
                    if q == M:
                        break
            if len(T) != M:
                raise
            cf0 = np.zeros_like(C)
            q = 0
            for i in range(nocc):
                if not i in T:
                    cf0[:,q] = C[:,i]
                    q += 1

    #        print(C[pz][:,list(T)])
    #        print(T)
    #        raise
            Cup = np.zeros((mol.nao,M))
            Cdn = np.zeros((mol.nao,M))
            for i in range(M//2):
                Cup[pz[carbons[2*i]],i] = 1
                Cdn[pz[carbons[2*i+1]],i] = 1
            for i in range(M//2):
                Cup[pz[carbons[2*i+1]],i+M//2] = 1
                Cdn[pz[carbons[2*i]],i+M//2] = 1

            S_ = Cup.T.dot(S).dot(Cup)
            val,vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
            Cup = Cup.dot(S_)

            S_ = Cdn.T.dot(S).dot(Cdn)
            val,vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
            Cdn = Cdn.dot(S_)

            D = C.copy()
            for i in range(M):
                C[:,T[i]] = Cup[:,i]
                D[:,T[i]] = Cdn[:,i]
            S_ = C.T.dot(S).dot(C)
            val,vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
            C = C.dot(S_)
            S_ = D.T.dot(S).dot(D)
            val,vec = sl.eigh(S_)
            S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
            D = D.dot(S_)
            mocc = np.zeros((mol.nao))
            mocc[:nocc] = 1
            mfu = scf.UHF(mol).newton()
            dm = mfu.make_rdm1((C,D),(mocc,mocc))
            mfu = mfu.run(dm)
    #            mo1 = mfu.stability()[0]
    #            dm = mfu.make_rdm1(mo1,mfu.mo_occ)
    #            mfu = mfu.run(dm)
            Cup,Cdn = mfu.mo_coeff

            "Build charge density and diagonalize for orbitals"
            chg_den = Cup[:,:nocc].dot(Cup[:,:nocc].T) + Cdn[:,:nocc].dot(Cdn[:,:nocc].T)
            chg_den = chg_den.dot(S)
            val,vec = sl.eig(chg_den)
            val = np.real(val)
            vec = np.real(vec)
            T = []
            V = []
            U = []
            for i in range(len(val)):
                if ((val[i] > 0.01) and (val[i] < 1.99)):
                    T.append(i)
                elif val[i] > 1.99:
                    V.append(i)
            T = np.array(T)
            if len(T) != M:
                raise
            if len(V) != ic0:
                raise
    #        Q = np.diag(vec[pz,:].T.dot(S[pz,:][:,pz]).dot(vec[pz,:]))
    #        T = np.where(Q>0.35)[0]
            Cac = vec[:,T]
            cf0 = vec[:,V]
            S_ = np.diag(1./np.sqrt(np.diag(cf0.T.dot(S).dot(cf0))))
            cf0 = cf0.dot(S_)
    #        print(df0.T.dot(S).dot(df0))
    #        qwd = df0.T.dot(S).dot(cf0[:,:ic0])
    #        val,vec = sl.eigh(qwd)
    #        print(val)
    #        raise


            if not unrestrict:
                "Make ideal orbitals"
                D = np.zeros((mol.nao,M))
                for i in range(M//2):
                    D[pz[carbons[2*i]],2*i] = 1
                    D[pz[carbons[2*i+1]],2*i] = 1
                    D[pz[carbons[2*i]],2*i+1] = 1
                    D[pz[carbons[2*i+1]],2*i+1] = -1
                S_ = D.T.dot(S).dot(D)
                val,vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
                D = D.dot(S_)

                "Project ideal orbitals onto active space"
                Dac  = Cac.dot(Cac.T).dot(S).dot(D)
                S_ = Dac.T.dot(S).dot(Dac)
                val,vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
                cf = Dac.dot(S_)
                print(cf[pz,:])
                U = np.eye(M)
            else:
    #            Pup = Cup[:,:nocc].dot(Cup[:,:nocc].T)
    #            Pdn = Cdn[:,:nocc].dot(Cdn[:,:nocc].T)
    #            val,vec = sl.eig(Pup.dot(S))
    #            val = np.real(val)
    #            vec = np.real(vec)
    #            Q = np.where(val>0.99)[0]
    #            print(vec[pz,:][:,Q])
    #            print(val)
    #            if len(Q) != nocc:
    #                raise
    #            di = np.diag(vec[:,Q][pz,:].T.dot(S[pz,:][:,pz]).dot(vec[:,Q][pz,:]))
                T = np.zeros((M//2),dtype=int)
                q = 0
                di = np.diag(Cup[:,:nocc][pz,:].T.dot(S[pz,:][:,pz]).dot(Cup[:,:nocc][pz,:]))
                for i in range(nocc):
                    if di[i] > 0.1:
                        T[q] = i
                        q += 1
                        if q == M//2:
                            break
                if q != M//2:
                    raise
                print(Cup[:,T][pz,:])
                V = np.zeros((M//2),dtype=int)
                q = 0
                di = np.diag(Cup[:,nocc:][pz,:].T.dot(S[pz,:][:,pz]).dot(Cup[:,nocc:][pz,:]))
                for i in range(mol.nao-nocc):
                    if di[i] > 0.1:
                        V[q] = i
                        q += 1
                        if q == M//2:
                            break
                if q != M//2:
                    raise
                V += nocc
                print(Cup[:,V][pz,:])
                Cup = np.hstack((Cup[:,T],Cup[:,V]))
                T = np.zeros((M//2),dtype=int)
                q = 0
                di = np.diag(Cdn[:,:nocc][pz,:].T.dot(S[pz,:][:,pz]).dot(Cdn[:,:nocc][pz,:]))
                for i in range(nocc):
                    if di[i] > 0.1:
                        T[q] = i
                        q += 1
                        if q == M//2:
                            break
                if q != M//2:
                    raise
                print(Cdn[:,T][pz,:])
                V = np.zeros((M//2),dtype=int)
                q = 0
                di = np.diag(Cdn[:,nocc:][pz,:].T.dot(S[pz,:][:,pz]).dot(Cdn[:,nocc:][pz,:]))
                for i in range(mol.nao-nocc):
                    if di[i] > 0.1:
                        V[q] = i
                        q += 1
                        if q == M//2:
                            break
                if q != M//2:
                    raise
                V += nocc
                print(Cdn[:,V][pz,:])
                Cdn = np.hstack((Cdn[:,T],Cdn[:,V]))
                Dup = np.zeros((mol.nao,M))
                Ddn = np.zeros((mol.nao,M))
                for i in range(M//2):
                    Dup[pz[carbons[2*i]],i] = 1
                    Ddn[pz[carbons[2*i+1]],i] = 1
                for i in range(M//2):
                    Dup[pz[carbons[2*i+1]],M//2+i] = 1
                    Ddn[pz[carbons[2*i]],M//2+i] = 1
                Dup  = Cup.dot(Cup.T).dot(S).dot(Dup)
                S_ = Dup.T.dot(S).dot(Dup)
                val,vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
                Dup = Dup.dot(S_)
                Ddn  = Cdn.dot(Cdn.T).dot(S).dot(Ddn)
                S_ = Ddn.T.dot(S).dot(Ddn)
                val,vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
                Ddn = Ddn.dot(S_)
                "Project ideal orbitals onto active space"
                Dac  = Cac.dot(Cac.T).dot(S).dot(Dup)
                S_ = Dac.T.dot(S).dot(Dac)
                val,vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
                cf = Dac.dot(S_)
                Dac  = Cac.dot(Cac.T).dot(S).dot(Ddn)
                S_ = Dac.T.dot(S).dot(Dac)
                val,vec = sl.eigh(S_)
                S_ = vec.dot(np.diag(1./np.sqrt(val))).dot(vec.T)
                cfdn = Dac.dot(S_)
                print(cf[pz,:])
                print(cfdn[pz,:])
                U = cfdn.T.dot(S).dot(cf)
                U = U.T
                print(atoms)

    elif molecule == -1:
        "Hubbard model"
        n = 7
        m = 4*n+2
        U = 2
    #    T = 1
        Hc = np.zeros((m,m))
        Hc[:-1,1:] -= np.eye(m-1)*T
        Hc += Hc.T
        Hc[0,-1] = Hc[-1,0] = -T
        eri = np.zeros((m,m,m,m))
        for i in range(m):
            eri[i,i,i,i] = U
        cf = np.eye(m)
        nb = m
        enr = 0.
        no = m

    else:
        "H2 cluster"
        n = 6
        mol = gto.Mole()
        mol.atom = []
        "Define molecule"
        mol.atom.append([1,(-0.7,-3.0,0.0)])
        mol.atom.append([1,(+0.7,-3.0,0.0)])
        mol.atom.append([1,(-0.7, 0.0,0.0)])
        mol.atom.append([1,(+0.7, 0.0,0.0)])
        if n > 4:
            mol.atom.append([1,(-0.7,+3.0,0.0)])
            mol.atom.append([1,(+0.7,+3.0,0.0)])
        if n > 6:
            mol.atom.append([1,(-0.7,+6.0,0.0)])
            mol.atom.append([1,(+0.7,+6.0,0.0)])
        if n > 8:
            mol.atom.append([1,(-0.7,+9.0,0.0)])
            mol.atom.append([1,(+0.7,+9.0,0.0)])
        mol.unit = 'au'
        mol.basis = 'sto-6g'
        mol.build()

        cf0 = np.zeros((n,n,))
        nb, no = cf0.shape
        cf0[0,0] = +1./np.sqrt(2.)
        cf0[1,0] = +1./np.sqrt(2.)
        cf0[0,1] = +1./np.sqrt(2.)
        cf0[1,1] = -1./np.sqrt(2.)

        if n > 2:
            cf0[2,2] = +1./np.sqrt(2.)
            cf0[3,2] = +1./np.sqrt(2.)
            cf0[2,3] = +1./np.sqrt(2.)
            cf0[3,3] = -1./np.sqrt(2.)

        if n > 4:
            cf0[4,4] = +1./np.sqrt(2.)
            cf0[5,4] = +1./np.sqrt(2.)
            cf0[4,5] = +1./np.sqrt(2.)
            cf0[5,5] = -1./np.sqrt(2.)
        if n > 6:
            cf0[6,6] = +1./np.sqrt(2.)
            cf0[7,6] = +1./np.sqrt(2.)
            cf0[6,7] = +1./np.sqrt(2.)
            cf0[7,7] = -1./np.sqrt(2.)
        if n > 8:
            cf0[8,8] = +1./np.sqrt(2.)
            cf0[9,8] = +1./np.sqrt(2.)
            cf0[8,9] = +1./np.sqrt(2.)
            cf0[9,9] = -1./np.sqrt(2.)

    #    mf = scf.RHF(mol).run()
        S = mol.intor_symmetric("int1e_ovlp")
        M = np.dot(cf0.T,np.dot(S,cf0))
        val, vec = sl.eigh(-M)
        U = np.dot(vec*1./np.sqrt(-val), vec.T)

        cf = np.dot(cf0,U)
        N,M = n,n
        U = np.eye(M)
    #    print(cf)

    if molecule > -1 or molecule == -2:
    #    "2eri over active orbtials only"
        print("Entering ints")
        eri = ao2mo.outcore.full_iofree(mol, cf, aosym='s1')
        eri = eri.reshape((no,)*4)
        print("eri done")

        Hc = mol.intor_symmetric('int1e_kin') \
           + mol.intor_symmetric('int1e_nuc')
        enr = mol.energy_nuc()
        print("have Hc")
    "remove core potential from hamiltonian(ic0=number of core)"
    if molecule > 0:
    #    mf = scf.RHF(mol)
        dm_core = np.dot(cf0[:,:ic0],cf0[:,:ic0].T)
        G = mf.get_veff(dm=dm_core*2)
        ecore  = 2*np.einsum('ij,ji->', Hc, dm_core)
        ecore +=   np.einsum('ij,ji->', G, dm_core)
    else:
        G = np.zeros((nb,nb,))
        ecore = 0.

    i1s = np.dot(cf.T, np.dot(Hc+G, cf))

    from sys import path
    path.append('/zfshomes/jrkeyes/cluster_fool/cluster')
    from write_ints import write_ints_
    import save_det

    write_ints_('tmp.ints',enr+ecore,i1s,eri)
    #save_det.write_ruhf ('tmp.det', np.eye(no), np.eye(no))
    save_det.write_ruhf ('tmp.det', np.eye(no), U)

    print("cMF Start")
    #cMF starts here
    #Convert integrals to GHF form
    cf_ = np.zeros((cf.shape[0],2*M))
    cf_[:,::2] = cf
    cf_[:,1::2] = cf
    cf = cf_.copy()
    i1s_ = np.zeros((2*M,)*2)
    i1s_[::2,::2] = i1s
    i1s_[1::2,1::2] = i1s
    i1s = i1s_.copy()
    eri = eri.transpose(0,2,1,3)
    eri_ = np.zeros((2*M,)*4)
    eri_[::2,::2,::2,::2] = eri
    eri_[1::2,1::2,1::2,1::2] = eri
    eri_[1::2,::2,1::2,::2] = eri
    eri_[::2,1::2,::2,1::2] = eri
    eri = eri_.copy()
    del i1s_,eri_


    #Orbs per cluster(assume half filled, Sz=0 mean field ,hard coded for Norb = 4)
    Norb = 4
    assert Norb == 4
    nclus = M*2//Norb
    nstat = 2**(Norb)

    #Build orbital to config mapping(annihilation op, transpose for creation)
    if False:
        Cmap = np.zeros((nstat,nstat,Norb,nclus))
        Cmap[8,0,0,0] = -1
        Cmap[9,1,0,0] = -1
        Cmap[2,11,0,0] = 1
        Cmap[3,13,0,0] = 1
        Cmap[7,4,0,0] = -1
        Cmap[5,12,0,0] = 1
        Cmap[10,6,0,0] = 1
        Cmap[14,15,0,0] = -1

        Cmap[6,0,1,0] = 1
        Cmap[1,12,1,0] = -1
        Cmap[7,2,1,0] = -1
        Cmap[3,14,1,0] = 1
        Cmap[4,11,1,0] = -1
        Cmap[9,5,1,0] = -1
        Cmap[10,8,1,0] = 1
        Cmap[13,15,1,0] = 1

        Cmap[0,11,2,0] = 1
        Cmap[1,13,2,0] = -1
        Cmap[8,2,2,0] = 1
        Cmap[9,3,2,0] = -1
        Cmap[6,4,2,0] = 1
        Cmap[5,14,2,0] = -1
        Cmap[10,7,2,0] = 1
        Cmap[12,15,2,0] = -1

        Cmap[0,12,3,0] = 1
        Cmap[6,1,3,0] = 1
        Cmap[2,14,3,0] = 1
        Cmap[7,3,3,0] = 1
        Cmap[4,13,3,0] = 1
        Cmap[8,5,3,0] = 1
        Cmap[10,9,3,0] = 1
        Cmap[11,15,3,0] = 1

    Cmap, part, Sz = Build_Cmap(Norb//2,nclus,Norb//4,Norb//4)
    print("Built Cmap")

    #Build initial guess(HF initial guess to start)
    D = np.zeros((nclus,nstat,nstat))
    D[0,:,:] = np.eye(nstat)
    for w in range(1,nclus):
        D[w] = D[0]
    print(nstat)
    print(D[0,:,0])
    print("Built guess")

    #Transform integrals to config basis
    i1s = i1s.reshape((nclus,Norb)*2).transpose(0,2,1,3)
    eri = eri.reshape((nclus,Norb,)*4).transpose(0,2,4,6,1,3,5,7)
    print("transformed integrals")
    anti = True
    if anti:
        eri -= eri.transpose(0,1,3,2,4,5,7,6)

    for w in range(1,nclus):
        Cmap[:,:,:,w] = Cmap[:,:,:,0]
    do_cMF = True
    if do_cMF:
        Energy,D,C,orben,i1s,eri = cMF_opt(i1s,eri,D,cf,thrsh_state=1e-9,thrsh_orb=1e-9,gnorm_orb=1.e-8)
        print("Final Energy: ",Energy+enr+ecore)
        print("iubqwdiuqwidubqwd")
    else:
        "Get data from carlos' code"
        with h5.File('outx.det') as fh5:
            X = fh5['/dmat'][()].T

        nb, no = X.shape
        no = no//2
        Xu = X[:,:no]
        Xd = X[:,no:]
        print(X)

        i1s, eri = update_ints(np.eye(nb),X,i1s,eri)

        D = np.zeros((nclus,nstat,nstat))
        for i in range(nclus):
            D[i] = np.eye(nstat)

        H1,H2,den = build_h0(i1s,eri,D)
        Energy,D,orben = cMF_state(i1s,eri,D,nstep=32,thrsh=1.e-6)
        print("Final Energy: ",Energy+enr+ecore)

    def newton(G,H):
        """Perform contraction between the inverse hessian and gradient
        to define search direction"""

        N = G[0].shape[0]
        def fun(x):
            x = x.reshape(len(G),N,N)

            out = np.zeros_like(x)
            k = 0
            l = 0
            for i in range(nclus):
                for j in range(i+1,nclus):
                    res = np.zeros((N,N))
                    for ii in range(nclus):
                        for jj in range(ii+1,nclus):
                            h = H[l]
                            res += np.einsum("cdab,cd->ab",h,x[k],optimize=True)
                            l += 1
                    out[k] = res.copy()
                    k += 1
            return out.flatten()
        b = np.zeros((len(G),N,N))
        for i in range(len(G)):
            b[i] = G[i]

        b = b.flatten()
        m = len(b)

        x, info = ssl.minres(ssl.LinearOperator((m,m), matvec=fun), b)

        x = x.reshape(len(G),N,N)
        out = []
        for i in range(len(G)):
            out.append(x[i])
        if info == 0:
            return out
        else:
            raise Exception("Minres alg did not converge")

    def cMF_orb_hess(i1s,eri,Cmap,D):
        """Calculate the orbital hessian"""

        if not anti:
            eri -= eri.transpose(0,1,3,2,4,5,7,6)
        Cmap = np.einsum("ijpw,wik,wjl->klpw",Cmap,D,D,optimize=True)

        H = []
        for i in range(nclus):
            for j in range(i+1,nclus):
                for k in range(nclus):
                    for l in range(k+1,nclus):
                        h = np.zeros((Norb,)*4)
                        h += cMF_orb_hess_ijkl(i,j,k,l,i1s,eri,Cmap)
                        h -= cMF_orb_hess_ijkl(j,i,k,l,i1s,eri,Cmap).transpose(1,0,2,3)
                        h -= cMF_orb_hess_ijkl(i,j,l,k,i1s,eri,Cmap).transpose(0,1,3,2)
                        h += cMF_orb_hess_ijkl(j,i,l,k,i1s,eri,Cmap).transpose(1,0,3,2)
                        h += cMF_orb_hess_ijkl(k,l,i,j,i1s,eri,Cmap).transpose(2,3,0,1)
                        h -= cMF_orb_hess_ijkl(k,l,j,i,i1s,eri,Cmap).transpose(3,2,0,1)
                        h -= cMF_orb_hess_ijkl(l,k,i,j,i1s,eri,Cmap).transpose(2,3,1,0)
                        h += cMF_orb_hess_ijkl(l,k,j,i,i1s,eri,Cmap).transpose(3,2,1,0)

                        H.append(h)
        return H

    def cMF_orb_grad(i1s,eri,D):
        """Calculate the orbital rotation gradient"""
        G = []
        for i in range(nclus):
            for j in range(i+1,nclus):
                g = cMF_orb_grad_nm(i,j,i1s,eri,D)
                G.append(g)
        return G

    def Coef_Z(C1,C2,Z):
        """Returns e^(Z)Coef"""
        n1, n2 = Z.shape
        assert C1.shape[1] == n1
        assert C2.shape[1] == n2
        if np.any(Z[::2,1::2]) or np.any(Z[1::2,::2]):
            raise Exception("Attempting to mix orbitals of different spin")

        L = sl.cholesky(np.eye(n1) + (Z.T).dot(Z), lower=True)
        L = sl.solve_triangular(L, np.eye(n1), lower=True)
        M = sl.cholesky(np.eye(n2) + Z.dot(Z.T), lower=True)
        M = sl.solve_triangular(M, np.eye(n2), lower=True)

        C1z = np.empty_like(C1)
        C2z = np.empty_like(C2)

        C1z = (C1 + C2.dot(Z)).dot(L.T)
        C2z = (C2 - C1.dot(Z.T)).dot(M.T)
        return C1z,C2z

    def Coef_Z_uni(C1,C2,Z):
        """Returns e^(Z)Coef using the unitary operator"""
        n1, n2 = Z.shape
        assert C1.shape[1] == n1
        assert C2.shape[1] == n2
        if np.any(Z[::2,1::2]) or np.any(Z[1::2,::2]):
            raise Exception("Attempting to mix orbitals of different spin")
        Z_ = np.zeros((n1+n2,)*2)
        Z_[n1:,:n1] = Z
        Z_[:n1,n1:] = -Z.T
        U = sl.expm(Z_)
        C = np.hstack((C1,C2))
        Cp = C.dot(U)
        return Cp[:,:n1],Cp[:,n1:]

    def update_orbs(C,Z):
        """Given a list of Z matrices, rotate orbitals"""
        k = 0
        C_ = C.copy()
        for n in range(nclus):
            for m in range(n+1,nclus):
                C1 = C_[:,Norb*n:Norb*(n+1)]
                C2 = C_[:,Norb*m:Norb*(m+1)]
                Z_ = Z[k]
                if np.any(Z_[::2,1::2]) or np.any(Z_[1::2,::2]):
                    raise Exception("Attempting to mix orbitals of different spin")
                k += 1
                C1z,C2z = Coef_Z_uni(C1,C2,Z_)
                C_[:,Norb*n:Norb*(n+1)] = C1z
                C_[:,Norb*m:Norb*(m+1)] = C2z
        return C_
# fmt: on
