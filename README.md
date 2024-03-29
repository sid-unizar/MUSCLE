# MUSCLE
## Building MUSCLE, a Dataset for MUltilingual Semantic Classification of Links between Entities
This repository contains the datasets, scripts and notebooks to reproduce the results of the paper *"Building MUSCLE, a Dataset for MUltilingual Semantic Classification of Links between Entities"*.

MUSCLE dataset contains five types of lexico-semantic relations (hypernymy, hyponymy, meronymy, holonymy and antonymy) and a random relation in 25 languages. The data was extracted from Wikidata, a large and highly multilingual factual Knowledge Graph. The covered languages and their ISO 639-1 codes are: Arabic (ar), Catalan (ca), Czech (cs), Danish (da), German (de), English (en), Spanish (es), Farsi (fa), Finnish (fi), French (fr), Hebrew (he), Hungarian (hu), Indonesian (id), Italian (it), Japanese (ja), Korean (ko), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Serbian (sr), Swedish (sv), Turkish (tr), Ukrainian (uk), and Chinese (zh).

The MUSCLE dataset can be downloaded (csv files):
- Random split (identified by RanS in the paper): [train dataset](https://drive.google.com/uc?id=1vXn30idOKST5vdfcW7AHHcGO0f2Piu_U&export=download) and [test dataset](https://drive.google.com/uc?id=13utdPI11VYjSpYOX5vIBJ4TH_1vF7hDP&export=download)
- Semantic split (identified by SemS in the paper): [train dataset](https://drive.google.com/uc?id=1-X3DZZu_GMd22T2jnvwiA39hPiaaHwgx&export=download) and [test dataset](https://drive.google.com/uc?id=1fVV2iObSLesVRnqfbrFbgCTR-_eESSJa&export=download)

Each line in the train/test datasets contains the following fields: 

|Field| Description|
|-------|------|
|`subject`| Wikidata concept id|
|`object`| Wikidata concept id|
|`relation_type`| One of hypernymy for, hyponymy for, meronymy for, holonymy for, antonymy for, and random|
|`property`| Wikidata property id or `random` label. If the id of a property is ended with `_inv`, the subject and object concepts in the original property in Wikidata have been inverted.|
|`xx_label_subject`| label for subject concept in language `xx`, where `xx` is a language ISO code|
|`xx_label_obbject`| label for object concept in language `xx`, where `xx` is a language ISO code|

For ease of use, we also provide the MUSCLE dataset in the following format (tsv files): each line contains subject label in a language, object label in the same language, relation type, ISO language code, concept Wikidata id for subject, concept Wikidata id for object and Wikidata property id.
- Random split: [train dataset](https://drive.google.com/uc?id=1MtwHxRgPQqTh4KWU_XoZVsAdPbIjyOp-&export=download) and [test dataset](https://drive.google.com/uc?id=1EAT_HnVSmkNDewcomRieMwiq7LmG7G63&export=download)
- Semantic split: [train dataset](https://drive.google.com/uc?id=1KzyEKdIG8wtd7126KpLttXRtboRRDk7L&export=download) and [test dataset](https://drive.google.com/uc?id=1UvgvBEuPBhFU-nEbZxvNbkI9p1ZsX5-V&export=download)

We give a brief description of the noteboooks, the scripts and other datasets used in the paper. See the mentioned Section in this document for a extended description:
- [Raw dataset](https://drive.google.com/uc?id=1RH8U3TGbUtSuSHk32byeQ2IJ4vn850WT&export=download): raw dataset from which MUSCLE is built (see Section 3 bellow).
- `louvainGlobalWithIDs.csv`: file containing the detected Louvain communities (Section 4 bellow).
- `complete_dataset_generator.ipynb`: notebook to generate a complete dataset (Section 3 bellow).
- `semantic_split_MUSCLE.ipynb`: notebook to generate the train/test datasets for the semantic split of MUSCLE (Section 5 bellow).
- `stats_muscle.ipynb`: some stats of the MUSCLE datasets (Section 5 bellow).
- `XL_lrc_train_evaluate.py`: Python script to fine-tune/evaluate the MUSCLE dataset (Section 6 bellow).
- `XL_train_evaluate_launcher.ipynb`: usage examples of the Python script (Section 5 bellow).
- `token_distribution_dataset.ipynb`: risk metrics (Section 6 bellow).

All the notebooks are fully functional in Google Colab.

### **Section 3. Data Design**
As explained in Section 3 of the paper, and after analyzing the Wikipedia concepts and properties that best fit the five lexico-semantic relationship considered, the starting point to build the MUSCE dataset is the [following *raw* dataset](https://drive.google.com/uc?id=1RH8U3TGbUtSuSHk32byeQ2IJ4vn850WT&export=download) (1.5 GB of uncompressed text). The structure of this file is similar to that the MUSCLE dataset, except for there is no hypernoym relation (the hyperonym relations are created later reversing some of the hyponym relations). There are $2.605.088$ of subject/concepts pairs and their regarding relationships in $40$ languages ($42$ languages, if we consider that for Chinese, we get also the traditional Chinese, zh-hant, and the simplified Chinese, zh-hans). For a specific language, the translation labels for the subject and the object concepts are both available or both not (marked in the file with `\N`). We consider that this raw dataset can be very useful to the community to create new datasets.

Since there are no available translation labels in all languages for each subject/object concepts, the first step to build the MUSCLE dataset is to obtain what we call *a complete dataset in N languages* from the raw dataset. That is, a dataset where the translation labels are available for $N$ languages. We set $N=25$ to get a trade-off between the number of concept pairs and the number of languages presented in MUSCLE. The notebook `complete_dataset_generator.ipynb` is used to create such a complete dataset. It is also pruned some pairs (the five filters applied in the notebook), reversed the 50% of the hyponym relations to obtain the hyperonym relations, and removed some random relation pairs so that the the random relations are the 70% of the total, as in similar datasets in the literature ([CogALex-V](https://aclanthology.org/W16-5309/)). Following the notation of the notebook, this process produce a file named `dataset_P_L25.csv`. This dataset is the basis to obtain the train/test datasets of the MUSCLE dataset.

### **Section 4. Dataset Analysis**
For the semantic split, we study the Louvain communities that appear using the undirect graph induced by the relations in the raw dataset (removing the random relations) with the graph database Neo4j. The obtained communities can be consulted in the file `louvainGlobalWithIDs.csv` contained in this repository.

### **Section 5. Dataset Configurations**
Once the file `dataset_P_L25.csv` and the Louvain communities are obtained, the final train/test datasets of MUSCLE are generated:
- For the random split, the subject/object pairs in `dataset_P_L25.csv` are randomly split (50% train / 50% test) stratified by the relation type. Note that `dataset_P_L25.csv` file is the union of the train and test dataset of the MUSCLE random split.
- For the semantic split, we split the dataset by semantic domains represented by the Louvain communities detected previously. We added whole communities to each split minimizing the amount of relationships between splits. Such relationships are discarded to isolate semantic splits. The notebook `semantic_split_MUSCLE.ipynb` contains the code to obtain the semantic split.

The stats of the MUSCLE dataset (number of concepts/relations in Table 4, multiwords in Figure 1, participation in Figure 2) can be obtained with notebook `stats_muscle.ipynb`.

### **Section 6. Dataset Evaluation**
**Subsections 6.1, 6.3, 6.4**: Fine-tuning experiments in these sections can be reproduced with the Python script `XL_lrc_train_evaluate.py`. Some usage examples of the the script are in the nootebook `XL_train_evaluate_launcher.ipynb`. The notebook `process_results_muscle.ipynb` can be used to process the result files produced by the script. In that notebook, our results are used, but it can be easily modified to process any other results (Tables 5, 7, 8, 12 and 13 in the paper).

**Subsection 6.2**: The tables containing the risk metrics (Tables 6, 9, 10 and 11) can be reproduced with the notebook `token_distribution_dataset.ipynb`.
