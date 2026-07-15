import os
import pickle

def load_complex_data(dataset_path):
    """
    遍历 dataset_path 下的每个复合物文件夹，
    读取其中的 <复合物名>_pocket.pdb、<复合物名>_ligand.sdf 以及 <复合物名>_protein.pdb 文件，
    返回一个以复合物名字为键、文件内容字典为值的字典。
    """
    data = {}
    # 遍历复合物文件夹
    for complex_folder in os.listdir(dataset_path):
        complex_path = os.path.join(dataset_path, complex_folder)
        if os.path.isdir(complex_path):
            # 初始化该复合物的字典
            data[complex_folder] = {}
            # 构造文件路径
            pocket_file = os.path.join(complex_path, f"{complex_folder}_pocket.pdb")
            ligand_file = os.path.join(complex_path, f"{complex_folder}_ligand.sdf")
            protein_file = os.path.join(complex_path, f"{complex_folder}_protein.pdb")
            # 读取文件内容（不做错误处理）
            with open(pocket_file, 'r') as f:
                data[complex_folder]['pocket'] = f.read()
            with open(ligand_file, 'r') as f:
                data[complex_folder]['ligand'] = f.read()
            with open(protein_file, 'r') as f:
                data[complex_folder]['protein'] = f.read()
    return data

def main():
    base_dir = "PDBbind_dataset"
    subsets = ["core-set", "refined-set"]
    for subset in subsets:
        subset_path = os.path.join(base_dir, subset)
        subset_data = load_complex_data(subset_path)
        # 输出文件命名为 core_set.pkl 和 refined_set.pkl（将 '-' 替换为 '_'）
        output_file = f"{subset.replace('-', '_')}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(subset_data, f)
        print(f"已保存 {subset} 数据到 {output_file}")

if __name__ == "__main__":
    main()
