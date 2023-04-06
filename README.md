# INFO: Intellectual and Friendly Dialogue Agents grounding
Source codes for the paper "[You Truly Understand What I Need: Intellectual and Friendly Dialogue Agents grounding Knowledge and Persona](https://aclanthology.org/2022.findings-emnlp.75.pdf)", accepted at EMNLP 2022 Findings.



## 1. Setup

### 1.1 Environmental Setup
The code runs with python 3.6.
All dependencies are listed in [requirements.txt](INFO_final/INFO/requirements.txt)

`pip install -r requirements.txt`

## 1.2 Dataset
You can download FoCus Dataset (Persona-Knowledge Chat) in [here](https://github.com/pkchat-focus/FoCus)

### 1.3 Create a knowledge index
Since we use RAG for dialogue generation, you need to create a knowledge index file for the generation.  
Before creating a knowledge index, you need to move Focus dataset into the `data/` folder.
```
|-- data
    |-- FoCus
        |-- train_focus.json
        `-- valid_focus.json
```

1\) The preprocessing code for creating raw knowledge is in the knowledge_index folder
```
create_knowledge_index_for_github.ipynb
```
2\) The code for creating a knowledge index file is as below
```
python use_own_knowledge_dataset --csv_path=your file --output_dir=your dir
```
or you can simply run sh file
```
sh create_knowldege_index.sh
```

we used the same file in the [transformers Github](https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/use_own_knowledge_dataset.py) but modified it a bit for preprocessing the raw knowledge


3\) After creating a knowledge index for FoCus Dataset, you should change your path in the `config/rag-tok-base-ct.json`

```
"data_dir": 
"save_dirpath": 
"knowledge_dataset_path": 
"knowledge_index_path": 
```



## 2. Training
Before you train the model, please modify the config file. 
```
sh train.sh
```

## 3. Evaluate
```
sh evaluate.sh
```

