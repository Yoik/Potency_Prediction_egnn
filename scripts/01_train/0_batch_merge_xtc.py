import subprocess
import os
import glob
import re
import sys
import yaml

# =========================================================
# 定位项目根目录并加载核心库
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上退两级，从 scripts/01_train 退到 egnn 根目录
egnn_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(egnn_dir)
sys.path.append(os.path.join(egnn_dir, "core_lib"))

# =========================================================

def natural_sort_key(s):
    """
    自然排序：确保 step7_2.xtc 排在 step7_10.xtc 前面
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def merge_replicate(replicate_dir):
    # 1. 寻找所有分段文件 (这里得到的是全路径/相对长路径)
    full_path_files = glob.glob(os.path.join(replicate_dir, "step7_*.xtc"))
    if not full_path_files: return

    # 2. 排序
    full_path_files.sort(key=natural_sort_key)
    
    # [FIX] 关键修改：只提取文件名，不带路径
    # 因为我们稍后会用 cwd 进入该目录，所以只需要文件名
    local_files = [os.path.basename(f) for f in full_path_files]
    
    # 3. 定义输出文件
    output_filename = "merged.xtc"
    output_full_path = os.path.join(replicate_dir, output_filename)
    
    # 如果已经存在，跳过
    if os.path.exists(output_full_path):
        # 检查文件大小，如果是空的或者是失败的产物，建议删除重跑
        if os.path.getsize(output_full_path) < 100:
            print(f"[Warn] Removing broken merged.xtc in {replicate_dir}")
            os.remove(output_full_path)
        else:
            print(f"[Skip] {replicate_dir} already has merged.xtc")
            return

    print(f"\n[Processing] Merging {len(local_files)} files in {replicate_dir} ...")
    
    # 4. 构造交互式输入 (0, c, c, ...)
    # 第一个文件输入 '0'，后续输入 'c'
    # 注意：trjcat 有时候不仅问 settime，可能还会问其他。
    # -settime 标志通常会触发交互。
    # 这里的输入序列对应文件数量：file1(0), file2(c), file3(c)...
    inputs = ["0"] + ["c"] * (len(local_files) - 1)
    input_str = "\n".join(inputs) + "\n"

    # 5. 调用 gmx trjcat
    # 注意：这里使用的是 local_files
    cmd = ["gmx", "trjcat", "-f"] + local_files + ["-o", output_filename, "-settime"]
    
    try:
        # cwd=replicate_dir 让我们“进入”该目录执行命令
        process = subprocess.Popen(
            cmd, 
            cwd=replicate_dir, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 发送输入并获取结果
        stdout, stderr = process.communicate(input=input_str)
        
        if process.returncode == 0:
            print(f"  -> Success")
        else:
            print(f"  -> Failed! Error log:\n{stderr}")
            # print(f"  -> STDOUT:\n{stdout}") # 调试用
            
    except FileNotFoundError:
        print("  [Error] 'gmx' command not found. Please load GROMACS environment first!")
        sys.exit(1)
    except Exception as e:
        print(f"  [Error] Python execution failed: {e}")

def main():
    # =========================================================
    # 动态加载 config.yaml 获取 MD 原始数据路径
    # =========================================================
    config_path = os.path.join(egnn_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 读取 ../data，并转化为绝对路径以确保万无一失
    relative_md_dir = config['paths'].get('md_data_dir', '../data')
    root_dir = os.path.abspath(os.path.join(egnn_dir, relative_md_dir))
    
    print(f">>> Starting Batch Merge...")
    print(f">>> Target Data Directory: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"[Error] Directory not found: {root_dir}")
        sys.exit(1)
        
    for root, dirs, files in os.walk(root_dir):
        if "step7_1.xtc" in files:
            merge_replicate(root)
            
    print("\n>>> All merging tasks finished.")

if __name__ == "__main__":
    main()