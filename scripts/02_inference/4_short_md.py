#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import warnings
import openmm
from openmm import app, unit, Platform
from openmm.app import PDBFile, Modeller, ForceField, Simulation
from pdbfixer import PDBFixer
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit.topology import Molecule
import MDAnalysis as mda
from modules.sequence_aligner import OffsetCalculator
from src.config import init_config
import rdkit.Chem as Chem

def separate_receptor(input_struct, db_path):
    """
    只提取受体部分，保存为临时 PDB
    """
    print(f"[Pre-process] Extracting receptor from: {input_struct}")
    try:
        if input_struct.endswith(".cif"):
            u = mda.Universe(input_struct, format='mmcif')
        else:
            u = mda.Universe(input_struct)
    except Exception as e:
        print(f"Error loading structure: {e}")
        return None

    aligner = OffsetCalculator(db_path)
    best_score = -1
    receptor_atoms = None
    
    # 智能识别受体链
    for seg in u.segments:
        prot = seg.atoms.select_atoms("protein and name CA")
        if len(prot) < 50: continue
        seq = "".join([aligner.three_to_one.get(r, 'X') for r in prot.resnames])
        for key, data in aligner.db.items():
            score = aligner.aligner.score(data["seq"], seq) / len(data["seq"])
            if score > best_score:
                best_score = score
                receptor_atoms = seg.atoms.select_atoms("protein")

    if not receptor_atoms or best_score < 0.4:
        print("[Error] Could not identify receptor chain.")
        return None
    
    fd_rec, rec_pdb = tempfile.mkstemp(suffix="_receptor.pdb")
    os.close(fd_rec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        receptor_atoms.write(rec_pdb)
        
    print(f"  Receptor extracted to: {rec_pdb}")
    return rec_pdb

def run_short_md_sdf(rec_pdb_path, sdf_path, output_prefix, steps=50000):
    print(f"[OpenMM] Starting Short MD (100ps) using SDF ligand...")
    
    # 1. 修复受体 (PDBFixer)
    print("  Fixing Receptor (PDBFixer)...")
    fixer = PDBFixer(filename=rec_pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    print("  Adding hydrogens to Receptor...")
    fixer.addMissingHydrogens(7.4)

    # 2. 加载配体 (OpenFF 直接读取 SDF)
    print(f"  Loading Ligand from {sdf_path}...")
    try:
        # OpenFF 读取 SDF
        off_mol = Molecule.from_file(sdf_path, allow_undefined_stereo=True)
        # [修改] 不在这里操作 residues，移到 RDKit 部分
    except Exception as e:
        print(f"[Error] Failed to load SDF file: {e}")
        return []

    # 3. 准备配体 PDB 用于合并 (OpenMM 需要 PDB 格式的拓扑)
    print("  Converting Ligand to OpenMM format...")
    # 使用 RDKit 后端导出 PDB
    rdmol = off_mol.to_rdkit()
    
    # [关键修复] 在 RDKit 对象上强制将残基名设为 LIG
    for atom in rdmol.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("LIG")
        info.SetResidueNumber(1)
        info.SetChainId("L")
        atom.SetPDBResidueInfo(info)
        
    fd, temp_lig_pdb = tempfile.mkstemp(suffix="_temp_lig.pdb")
    os.close(fd)
    Chem.MolToPDBFile(rdmol, temp_lig_pdb)
    
    ligand_pdb = PDBFile(temp_lig_pdb)

    # 4. 合并受体和配体
    print("  Merging Receptor and Ligand...")
    try:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.add(ligand_pdb.topology, ligand_pdb.positions)
    except ValueError as e:
        print(f"[Error] Merging failed: {e}")
        return []

    # 5. 准备力场 (GAFF)
    print("  Parametrizing Ligand (GAFF from SDF)...")
    try:
        # 直接用 SDF 生成的 molecule 对象
        gaff = GAFFTemplateGenerator(molecules=[off_mol])
        
        forcefield = ForceField('amber14-all.xml', 'implicit/obc2.xml')
        forcefield.registerTemplateGenerator(gaff.generator)
    except Exception as e:
        print(f"[Error] ForceField setup failed: {e}")
        return []

    # 6. 创建系统
    print("  Creating OpenMM System...")
    try:
        system = forcefield.createSystem(modeller.topology, 
                                       nonbondedMethod=app.CutoffNonPeriodic,
                                       nonbondedCutoff=1.0*unit.nanometer,
                                       constraints=app.HBonds)
    except Exception as e:
        print(f"[Error] createSystem failed: {e}")
        return []

    # 7. 固定骨架
    print("  Applying Backbone Restraints...")
    force = openmm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", 100.0*unit.kilojoules_per_mole/unit.nanometer**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    
    for atom in modeller.topology.atoms():
        if atom.name == 'CA':
            force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    # 8. 模拟
    integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    
    # 平台选择
    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')
    
    print(f"  Using Platform: {platform.getName()}")
        
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print("  Minimizing Energy...")
    simulation.minimizeEnergy()

    saved_frames = []
    simulation.step(5000) # Pre-equil
    
    print(f"  Running Production MD ({steps} steps)...")
    frame_interval = steps // 10
    
    for i in range(10):
        simulation.step(frame_interval)
        state = simulation.context.getState(getPositions=True)
        frame_name = f"{output_prefix}_frame_{i}.pdb"
        with open(frame_name, 'w') as f:
            PDBFile.writeFile(simulation.topology, state.getPositions(), f)
        saved_frames.append(frame_name)
    
    if os.path.exists(temp_lig_pdb): os.remove(temp_lig_pdb)
    
    print(f"[Success] Generated {len(saved_frames)} frames.")
    return saved_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True, help="Boltz complex (for Receptor)")
    parser.add_argument("--sdf", required=True, help="Ligand SDF file (with correct 3D coords)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    config = init_config()
    db_path = os.path.join("data/gpcr_db.yaml")

    # 1. 提取受体
    rec_pdb = separate_receptor(args.structure, db_path)
    if not rec_pdb: sys.exit(1)

    # 2. 跑短 MD
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
    prefix = os.path.join(args.out_dir, args.name)
    
    frames = run_short_md_sdf(rec_pdb, args.sdf, prefix)
    
    if os.path.exists(rec_pdb): os.remove(rec_pdb)

if __name__ == "__main__":
    main()