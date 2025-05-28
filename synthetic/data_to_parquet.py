import os
import datasets
import copy

def main():
    combine=True
    base_dir = "./data/parquet_data"
    os.makedirs(base_dir, exist_ok=True)
    #"""
    argss=[]
    for n_nodes in [10, 20, 50, 100]:
        for n_actions in [5]:#, 10, 20]:
            for n_ablate in ["0.2"]:#"0.1", "0.2", "0.3"]:
                if type(n_ablate) == str:
                    n_ablate = int(float(n_ablate) * n_nodes * n_actions)
                for seed in [0]:#, 1, 2]:
                    argss.append({
                        "n_nodes": n_nodes,
                        "n_actions": n_actions,
                        "n_ablate": n_ablate,
                        "seed": seed
                    })
    #"""

    for args in argss:
        n_nodes = args["n_nodes"]
        n_actions = args["n_actions"]
        n_ablate = args["n_ablate"]
        seed = args["seed"]

        dataset_name = f"cfpark00/toy-multistep-nn_{n_nodes}-na_{n_actions}-nab_{n_ablate}-seed_{seed}"
        print(f"Processing dataset: {dataset_name}")

        try:
            ds = datasets.load_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
        dataset_dir = os.path.join(
                            base_dir,
                            f"nn_{n_nodes}",
                            f"na_{n_actions}",
                            f"nab_{n_ablate}",
                            f"seed_{seed}"
                        )
        os.makedirs(dataset_dir, exist_ok=True)
        for key in ds.keys():
            ds_ = ds[key]
            ds_=ds_.map(lambda x: {"raw_prompt": x["prompts"],'reward_model':{"ground_truth": x["completions"]},"data_source":"cfpark00/toy-multistep-reasoning"})
            file_path= os.path.join(dataset_dir, f"{key}.parquet")
            print(f"Saving test split to {file_path}")
            ds_.to_parquet(file_path)
        
        combined_keyss={
            "rl_nm_1_2_3_4": ["rl_nm_1","rl_nm_2","rl_nm_3","rl_nm_4"],
            "test_nm_1_2_3_4": ["test_nm_1","test_nm_2","test_nm_3","test_nm_4"],
        }
        for name,combined_keys in combined_keyss.items():
            if all([k in ds.keys() for k in combined_keys]):
                print("Making combined",name)
                # Combine the datasets
                ds_=datasets.concatenate_datasets([ds[k] for k in combined_keys])
                ds_=ds_.map(lambda x: {"raw_prompt": x["prompts"],'reward_model':{"ground_truth": x["completions"]},"data_source":"cfpark00/toy-multistep-reasoning"})
                file_path= os.path.join(dataset_dir, f"{name}.parquet")
                print(f"Saving test split to {file_path}")
                ds_.to_parquet(file_path)
                
def main_v2():
    combine=True
    base_dir = "./data/parquet_data_v2"
    os.makedirs(base_dir, exist_ok=True)
    #"""
    argss=[]
    for n_nodes in [20]:
        for n_actions in [10]:#, 10, 20]:
            for n_ablate in ["0.2"]:#"0.1", "0.2", "0.3"]:
                if type(n_ablate) == str:
                    n_ablate = int(float(n_ablate) * n_nodes * n_actions)
                for p in [0.90,0.95,1.00]:
                    for seed in [0]:#, 1, 2]:
                        argss.append({
                            "n_nodes": n_nodes,
                            "n_actions": n_actions,
                            "n_ablate": n_ablate,
                            "p": p,
                            "seed": seed
                        })
    #"""

    for args in argss:
        n_nodes = args["n_nodes"]
        n_actions = args["n_actions"]
        n_ablate = args["n_ablate"]
        p = args["p"]
        seed = args["seed"]

        dataset_name = f"cfpark00/toy-multistep-v2-nn_{n_nodes}-na_{n_actions}-nab_{n_ablate}-p_{int(p*100)}-seed_{seed}"
        print(f"Processing dataset: {dataset_name}")

        try:
            ds = datasets.load_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
        dataset_dir = os.path.join(
                            base_dir,
                            f"nn_{n_nodes}",
                            f"na_{n_actions}",
                            f"nab_{n_ablate}",
                            f"p_{int(p*100)}",
                            f"seed_{seed}"
                        )
        os.makedirs(dataset_dir, exist_ok=True)
        for key in ds.keys():
            ds_ = ds[key]
            ds_=ds_.map(lambda x: {"raw_prompt": x["prompt"],'reward_model':{"ground_truth": x["completion"]},"data_source":"cfpark00/toy-multistep-reasoning"})
            file_path= os.path.join(dataset_dir, f"{key}.parquet")
            print(f"Saving test split to {file_path}")
            ds_.to_parquet(file_path)
        
        combined_keyss={
            "train_rl": ["train_rl"],
            "test_nm_0_1_2_3_4": ["test_nm_0","test_nm_1","test_nm_2","test_nm_3","test_nm_4"],
        }
        for name,combined_keys in combined_keyss.items():
            if all([k in ds.keys() for k in combined_keys]):
                print("Making combined",name)
                # Combine the datasets
                ds_=datasets.concatenate_datasets([ds[k] for k in combined_keys])
                ds_=ds_.map(lambda x: {"raw_prompt": x["prompt"],'reward_model':{"ground_truth": x["completion"]},"data_source":"cfpark00/toy-multistep-reasoning"})
                file_path= os.path.join(dataset_dir, f"{name}.parquet")
                print(f"Saving test split to {file_path}")
                ds_.to_parquet(file_path)

def main_vs(version,split):
    base_dir = f"./data/parquet_data_v{version}_{split}"
    os.makedirs(base_dir, exist_ok=True)

    dataset_name = f"cfpark00/toy-multistep-v{version}-{split}"
    print(f"Processing dataset: {dataset_name}")

    try:
        ds = datasets.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
    dataset_dir = base_dir
    for key in ds.keys():
        ds_ = ds[key]
        ds_=ds_.map(lambda x: {"raw_prompt": x["prompt"],'reward_model':{"ground_truth": x["completion"]},"data_source":"cfpark00/toy-multistep-reasoning"})
        file_path= os.path.join(dataset_dir, f"{key}.parquet")
        print(f"Saving test split to {file_path}")
        ds_.to_parquet(file_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="5")
    parser.add_argument("--split", type=str, default="1")
    args = parser.parse_args()
    version=args.version
    split=args.split
    main_vs(version,split)