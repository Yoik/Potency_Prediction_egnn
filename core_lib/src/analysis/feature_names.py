from src.config import init_config

def get_feature_names():
    config = init_config()
    n_atoms = config.get_int("data.n_atoms")
    atom_feat_dim = config.get_int("data.atom_feat_dim")

    PHE_LABELS = config.get_list("residues.phe_residues")
    p1 = PHE_LABELS[0]
    p2 = PHE_LABELS[1]

    names = []

    # 原子特征
    for i in range(n_atoms):
        names.extend([
            f"Atom{i}_{p1}_Dist",
            f"Atom{i}_{p1}_Angle",
            f"Atom{i}_{p2}_Dist",
            f"Atom{i}_{p2}_Angle",
            f"Atom{i}_{p1}_Score",
            f"Atom{i}_{p2}_Score",
        ])

    # 全局特征
# 全局特征
    names.extend([
        "Global_Cos_Angle",

        # 6.51 / 6.52 各自统计
        f"{p1}_Global_Sum", f"{p1}_Global_Max", f"{p1}_Global_Norm",
        f"{p2}_Global_Sum", f"{p2}_Global_Max", f"{p2}_Global_Norm",

        # === 新增：6.51 vs 6.52 偏向性（路线 A 核心）===
        f"Delta_{p1}_{p2}_Global_Sum",
        f"Delta_{p1}_{p2}_Global_Max",
        f"Delta_{p1}_{p2}_Global_Norm",

        # 方向与电子形状
        "Lig_H6_Orientation",
        "ELF_H6_Axis_Cos",
        "ELF_Anisotropy",
    ])

    return names
