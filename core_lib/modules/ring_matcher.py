"""
ring_matcher.py (Enhanced with Substituent Vector Alignment)
环形匹配模块：支持苯环、吲哚、呋喃，并利用取代基防止翻转
"""

import numpy as np
import itertools
from scipy.spatial.distance import cdist

class RingMatcher:
    # ... (__init__ 和 _detect_all_rings 等保持不变，直到 _align_by_fingerprint) ...
    def __init__(self, ref_coords, ref_elements):
        self.ref_coords = ref_coords
        self.ref_elements = ref_elements
        self.n_ref = len(ref_coords)
        
        # 检测所有可能的环
        self.rings = self._detect_all_rings()
        
        if not self.rings:
            raise ValueError("No aromatic rings (6-membered or 5-membered) found")
        
        self.ref_ring_idx = self.rings[0]['indices']
        self.ring_type = self.rings[0]['type']
        self.ring_elements = [ref_elements[i] for i in self.ref_ring_idx]
        
        self._find_neighboring_atoms()

    def _get_substituent_vector(self, ring_atom_idx, all_coords, ring_indices):
        ring_pos = all_coords[ring_atom_idx]
        sub_vectors = []
        
        # 将 1.7 改为 2.2，覆盖 Cl, S, Br 等
        dists = np.linalg.norm(all_coords - ring_pos, axis=1)
        neighbors = np.where((dists > 0.1) & (dists < 2.2))[0]
        
        for n_idx in neighbors:
            if n_idx not in ring_indices:
                v = all_coords[n_idx] - ring_pos
                sub_vectors.append(v)
        
        if not sub_vectors: return np.zeros(3)
        avg_vec = np.mean(sub_vectors, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm > 0.001: return avg_vec / norm
        return np.zeros(3)

    def _align_by_fingerprint(self, candidate_atoms, full_md_atoms):
        """
        加强版指纹对齐 (v2.1 - 修复卤素/硫键长问题)
        """
        ref_indices = self.ref_ring_idx
        ref_coords = self.ref_coords[ref_indices]
        ref_elems = self.ring_elements
        
        cand_coords = candidate_atoms.positions
        cand_elems = [a.name[0] for a in candidate_atoms]
        
        # --- 计算参考体系的取代基掩码 ---
        ref_subs_mask = []
        dmat_ref_all = cdist(self.ref_coords, self.ref_coords)
        
        for r_idx in ref_indices:
            is_sub = False
            # 【修改点 2】阈值 1.7 -> 2.2
            neighbors = np.where((dmat_ref_all[r_idx] < 2.2) & (dmat_ref_all[r_idx] > 0.1))[0]
            for n in neighbors:
                if n not in ref_indices:
                    # 排除氢
                    if self.ref_elements[n].upper() != 'H':
                        is_sub = True
                        break
            ref_subs_mask.append(is_sub)

        # --- 计算 MD 候选体系的取代基掩码 ---
        md_subs_mask = []
        cand_global_indices = [a.index for a in candidate_atoms]
        all_ligand_pos = full_md_atoms.positions
        dmat_md_local = cdist(cand_coords, all_ligand_pos)
        
        for i in range(len(candidate_atoms)):
            is_sub = False
            # 【修改点 3】阈值 1.7 -> 2.2
            neighbors = np.where((dmat_md_local[i] < 2.2) & (dmat_md_local[i] > 0.1))[0]
            for n_idx in neighbors:
                atom = full_md_atoms[n_idx]
                if atom.index not in cand_global_indices:
                    name = atom.name.upper()
                    # 排除氢 (支持 H, HA, HB, 1H 等写法)
                    if not (name.startswith('H') or (name[0].isdigit() and 'H' in name)):
                         is_sub = True
                         break
            md_subs_mask.append(is_sub)

        # --- 匹配逻辑 ---
        dmat_ref_internal = cdist(ref_coords, ref_coords)
        dmat_cand_internal = cdist(cand_coords, cand_coords)
        
        mapping_dict = {} 
        used_indices = set()
        
        # 优先级排序
        priorities = []
        for i in range(len(ref_indices)):
            score = 0
            if ref_elems[i] != 'C': score += 20
            if ref_subs_mask[i]: score += 50 
            priorities.append((score, i))
        
        priorities.sort(key=lambda x: x[0], reverse=True)
        sorted_ref_indices = [p[1] for p in priorities]
        
        for i in sorted_ref_indices:
            r_e = ref_elems[i]
            r_has_sub = ref_subs_mask[i]
            r_dists = np.sort(dmat_ref_internal[i])
            
            best_j = -1
            min_score = float('inf')
            
            for j in range(len(candidate_atoms)):
                if j in used_indices: continue
                
                if cand_elems[j] != r_e: continue
                
                # 取代基状态匹配
                if r_has_sub != md_subs_mask[j]: 
                    penalty_sub = 1000.0 
                else:
                    penalty_sub = 0.0
                
                c_dists = np.sort(dmat_cand_internal[j])
                dist_score = np.sum(np.abs(r_dists - c_dists))
                
                total_score = dist_score + penalty_sub
                
                if total_score < min_score:
                    min_score = total_score
                    best_j = j
            
            # 【新增】调试信息：如果匹配失败，打印原因
            if best_j == -1 or min_score > 500.0: 
                # print(f"DEBUG: Match failed for Atom {i} ({r_e}). Min Score: {min_score}")
                # print(f"DEBUG: Ref Sub: {r_has_sub}, Possible MD Subs: {md_subs_mask}")
                return None
            
            mapping_dict[i] = best_j
            used_indices.add(best_j)
            
        final_mapping = [mapping_dict[i] for i in range(len(ref_indices))]
        
        # --- 手性检查 ---
        idx0, idx1 = final_mapping[0], final_mapping[1]
        dist_md_01 = np.linalg.norm(cand_coords[idx0] - cand_coords[idx1])
        dist_ref_01 = np.linalg.norm(ref_coords[0] - ref_coords[1])
        
        if abs(dist_md_01 - dist_ref_01) > 0.5:
            return None

        return final_mapping

    def _detect_all_rings(self):
        rings = []
        benzenes = self._detect_benzene_rings()
        if not benzenes: return rings
        fused = self._detect_fused_rings()
        if fused: return fused
        return benzenes

    def _detect_fused_rings(self):
        fused_rings = []
        try:
            six_rings = self._find_all_6rings()
            five_rings = self._find_all_5rings()
            if not six_rings or not five_rings: return fused_rings
            for six_ring in six_rings:
                six_set = set(six_ring['indices'])
                for five_ring in five_rings:
                    five_set = set(five_ring['indices'])
                    shared = six_set & five_set
                    if len(shared) == 2:
                        fused_indices = list(six_set | five_set)
                        fused_coords = self.ref_coords[fused_indices]
                        fused_elems = [self.ref_elements[i] for i in fused_indices]
                        five_ring_elems = [self.ref_elements[i] for i in five_ring['indices']]
                        if 'S' in five_ring_elems: ring_type = 'thiophene'
                        elif 'O' in five_ring_elems: ring_type = 'furan'
                        elif 'N' in five_ring_elems: ring_type = 'indole'
                        else: ring_type = None
                        if ring_type:
                            fused_rings.append({
                                'indices': fused_indices, 'type': ring_type, 'size': len(fused_indices),
                                'six_ring': six_ring['indices'], 'five_ring': five_ring['indices'],
                                'shared_atoms': list(shared), 'elements': fused_elems, 'coords': fused_coords
                            })
        except Exception: pass
        return fused_rings

    def _find_all_6rings(self):
        rings = []
        c_indices = [i for i, e in enumerate(self.ref_elements) if e == 'C']
        if len(c_indices) < 6: return rings
        c_coords = self.ref_coords[c_indices]
        dmat = cdist(c_coords, c_coords)
        adj = np.logical_and(dmat > 1.1, dmat < 1.7)
        for comb in itertools.combinations(range(len(c_indices)), 6):
            sub_idx = list(comb)
            curr_coords = c_coords[sub_idx]
            centered = curr_coords - curr_coords.mean(0)
            _, s, _ = np.linalg.svd(centered)
            if s[2] > 0.3: continue
            sub_adj = adj[np.ix_(sub_idx, sub_idx)]
            if not np.all(np.sum(sub_adj, axis=1) >= 2): continue
            ordered_indices = self._order_ring_indices(sub_idx, sub_adj)
            global_indices = [c_indices[i] for i in ordered_indices]
            ordered_local_indices = [sub_idx.index(i) for i in ordered_indices]
            rings.append({'indices': global_indices, 'type': 'benzene', 'size': 6, 'coords': curr_coords[ordered_local_indices]})
        return rings

    def _find_all_5rings(self):
        rings = []
        all_indices = list(range(self.n_ref))
        for comb in itertools.combinations(all_indices, 5):
            ring_idx = list(comb)
            ring_coords = self.ref_coords[ring_idx]
            ring_elems = [self.ref_elements[i] for i in ring_idx]
            has_hetero = any(elem in ring_elems for elem in ['N', 'O', 'S'])
            if not has_hetero: continue
            sub_dmat = cdist(ring_coords, ring_coords)
            sub_adj = np.logical_and(sub_dmat > 1.1, sub_dmat < 1.9)
            if not np.all(np.sum(sub_adj, axis=1) >= 1): continue
            try:
                ordered_indices = self._order_ring_indices_5_flexible(ring_idx, sub_adj)
                if ordered_indices is None: continue
                if 'S' in ring_elems: rt = 'thiophene'
                elif 'O' in ring_elems: rt = 'furan'
                elif 'N' in ring_elems: rt = 'pyrrole'
                else: continue
                rings.append({'indices': [ring_idx[i] for i in ordered_indices], 'type': rt, 'size': 5, 'elements': ring_elems, 'coords': ring_coords[ordered_indices]})
            except: continue
        return rings

    def _detect_benzene_rings(self): return self._find_all_6rings()
    
    def _order_ring_indices(self, indices, sub_adj):
        ordered = [indices[0]]; current = 0; used = {0}
        for _ in range(5):
            for n in np.where(sub_adj[current])[0]:
                if n not in used: ordered.append(indices[n]); used.add(n); current = n; break
        return ordered

    def _order_ring_indices_5(self, indices, sub_adj):
        ordered = [0]; current = 0; used = {0}
        for _ in range(4):
            for n in np.where(sub_adj[current])[0]:
                if n not in used: ordered.append(n); used.add(n); current = n; break
        return ordered

    def _order_ring_indices_5_flexible(self, indices, sub_adj):
        for start in range(len(indices)):
            ordered = [start]; current = start; used = {start}
            for _ in range(4):
                neighbors = np.where(sub_adj[current])[0]
                found_next = False
                for n in neighbors:
                    if n not in used: ordered.append(n); used.add(n); current = n; found_next = True; break
                if not found_next: break
            if len(ordered) == 5 and sub_adj[current, start]: return ordered
        return None

    def _find_neighboring_atoms(self):
        ring_set = set(self.ref_ring_idx)
        self.ref_neigh_idx = []
        dmat = cdist(self.ref_coords, self.ref_coords)
        for i in range(self.n_ref):
            if i not in ring_set and np.min(dmat[i, self.ref_ring_idx]) < 2.0:
                self.ref_neigh_idx.append(i)

    def match(self, md_atoms, anchor_com):
        if self.ring_type == 'benzene': return self._match_benzene(md_atoms, anchor_com)
        elif self.ring_type in ['indole', 'furan', 'benzofuran', 'thiophene']: return self._match_fused_system(md_atoms, anchor_com)            
        else: return None, None, None
        
    def _match_benzene(self, md_atoms, anchor_com):
        md_coords = md_atoms.positions
        md_c_indices = [i for i, a in enumerate(md_atoms) if a.name.startswith('C')]
        if len(md_c_indices) < 6: return None, None, None
        md_c_coords = md_coords[md_c_indices]
        dmat = cdist(md_c_coords, md_c_coords)
        adj = np.logical_and(dmat > 1.1, dmat < 1.70)
        found_rings = []; seen = set()
        def dfs(s, c, p):
            if len(p) == 6: return p if adj[c, s] else None
            for n in np.where(adj[c])[0]:
                if n == s and len(p) < 5: continue
                if n not in p:
                    r = dfs(s, n, p + [n])
                    if r: return r
            return None
        for i in range(len(md_c_indices)):
            res = dfs(i, i, [i])
            if res:
                s = tuple(sorted(res))
                if s not in seen: found_rings.append(list(res)); seen.add(s)
        if not found_rings: return None, None, None
        best_ring_local = None; min_dist = float('inf')
        for r in found_rings:
            g = [md_c_indices[k] for k in r]
            cent = md_coords[g].mean(0)
            d = np.linalg.norm(cent - anchor_com)
            if d < min_dist: min_dist = d; best_ring_local = g
        if best_ring_local is None: return None, None, None
        
        candidate_atoms = md_atoms[best_ring_local]
        # 【关键】传入 full_md_atoms=md_atoms，以便在指纹对齐时搜索取代基
        sorted_order = self._align_by_fingerprint(candidate_atoms, md_atoms)
        if sorted_order is None: return None, None, None
        
        md_ring_indices = [best_ring_local[i] for i in sorted_order]
        matched_atoms = md_atoms[md_ring_indices]
        ref_c_idxs = [i for i, e in enumerate(self.ref_elements) if e == 'C']
        cube_idxs = [{idx: rank for rank, idx in enumerate(ref_c_idxs)}[i] for i in self.ref_ring_idx if self.ref_elements[i] == 'C']
        return matched_atoms, cube_idxs, md_ring_indices

    def _match_fused_system(self, md_atoms, anchor_com):
        md_coords = md_atoms.positions
        heavy_mask = [a.name[0] in ['C', 'N', 'O', 'S'] for a in md_atoms]
        heavy_indices_local = [i for i, x in enumerate(heavy_mask) if x]
        if len(heavy_indices_local) < 9: return None, None, None
        heavy_coords = md_coords[heavy_indices_local]
        dmat = cdist(heavy_coords, heavy_coords)
        adj = np.logical_and(dmat > 1.1, dmat < 1.9) 
        def find_rings(target_len):
            found = []; seen = set()
            def dfs(s, c, p):
                if len(p) == target_len: return p if adj[c, s] else None
                for n in np.where(adj[c])[0]:
                    if n == s and len(p) < target_len - 1: continue
                    if n not in p:
                        r = dfs(s, n, p + [n])
                        if r: return r
                return None
            for i in range(len(heavy_indices_local)):
                if np.sum(adj[i]) >= 2:
                    res = dfs(i, i, [i])
                    if res:
                        s = tuple(sorted(res))
                        if s not in seen: found.append(set(res)); seen.add(s)
            return found
        rings_6 = find_rings(6); rings_5 = find_rings(5)
        if not rings_6 or not rings_5: return None, None, None
        valid_candidates = []
        for r6 in rings_6:
            for r5 in rings_5:
                shared = r6.intersection(r5)
                if len(shared) == 2:
                    fused_set = r6.union(r5)
                    if len(fused_set) != 9: continue
                    current_group_local = list(fused_set)
                    real_indices = [heavy_indices_local[i] for i in current_group_local]
                    cent = md_coords[real_indices].mean(0)
                    dist = np.linalg.norm(cent - anchor_com)
                    valid_candidates.append((dist, real_indices))
        valid_candidates.sort(key=lambda x: x[0])
        for dist, best_indices_local in valid_candidates:
            candidate_atoms = md_atoms[best_indices_local]
            sorted_local_order = self._align_by_fingerprint(candidate_atoms, md_atoms)
            if sorted_local_order is not None:
                final_md_indices = [best_indices_local[i] for i in sorted_local_order]
                matched_atoms = md_atoms[final_md_indices]
                ref_heavy_idxs = [i for i, e in enumerate(self.ref_elements) if e in ['C', 'N', 'O', 'S']]
                cube_idxs = [{idx: rank for rank, idx in enumerate(ref_heavy_idxs)}[i] for i in self.ref_ring_idx if i in ref_heavy_idxs]
                return matched_atoms, cube_idxs, final_md_indices
        return None, None, None