# MUSCLE
## Building MUSCLE, a Dataset for MUltilingual Semantic Classification of Links between Entities
This repository contains the datasets, scripts and notebooks to reproduce the results of the submitted paper *"Building MUSCLE, a Dataset for MUltilingual Semantic Classification of Links between Entities"*.

MUSCLE dataset contains five types of lexico-semantic relations (hypernymy, hyponymy, meronymy, holonymy and antonymy) and a random relation in 25 languages. The data was extracted from Wikidata, a large and highly multilingual factual Knowledge Graph. The covered languages and their ISO 639-1 codes are: Arabic (ar), Catalan (ca), Czech (cs), Danish (da), German (de), English (en), Spanish (es), Farsi (fa), Finnish (fi), French (fr), Hebrew (he), Hungarian (hu), Indonesian (id), Italian (it), Japanese (ja), Korean (ko), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Serbian (sr), Swedish (sv), Turkish (tr), Ukrainian (uk), and Chinese (zh).

The MUSCLE dataset can be download (csv files):
- Random split (identified by RanS in the paper): [train dataset](https://drive.google.com/uc?id=1vXn30idOKST5vdfcW7AHHcGO0f2Piu_U&export=download) and [test dataset](https://drive.google.com/uc?id=13utdPI11VYjSpYOX5vIBJ4TH_1vF7hDP&export=download)
- Semantic split (identified by SemS in the paper): [train dataset](https://drive.google.com/uc?id=1-X3DZZu_GMd22T2jnvwiA39hPiaaHwgx&export=download) and [test dataset](https://drive.google.com/uc?id=1fVV2iObSLesVRnqfbrFbgCTR-_eESSJa&export=download)

Each line in the train/test datasets contain the following fields: 

|Field| Description|
|-------|------|
|`subject`| Wikidata concept id|
|`object`| Wikidata concept id|
|`relation_type`| One of hypernymy for, hyponymy for, meronymy for, holonymy for, antonymy for, and random|
|`property`| Wikidata property id or `random` label. If the id of a property is ended with `_inv`, the subject and object concepts in the original property in Wikidata have been inverted.|
|`xx_label_subject`| label for subject concept in language `xx`, where `xx` is a language ISO code|
|`xx_label_obbject`| label for object concept in language `xx`, where `xx` is a language ISO code|

For ease of use of the datasets, we also provide them in the following format: each line contains subject label in a language, object label in the same language, relation type, ISO language code, concept Wikidata id for subject, concept Wikidata id for object and Wikidata property id.
- Random split: [train dataset](https://drive.google.com/uc?id=1MtwHxRgPQqTh4KWU_XoZVsAdPbIjyOp-&export=download) and [test dataset](https://drive.google.com/uc?id=1EAT_HnVSmkNDewcomRieMwiq7LmG7G63&export=download)
- Semantic split: [train dataset](https://drive.google.com/uc?id=1KzyEKdIG8wtd7126KpLttXRtboRRDk7L&export=download) and [test dataset](https://drive.google.com/uc?id=1UvgvBEuPBhFU-nEbZxvNbkI9p1ZsX5-V&export=download)

###**Section 3



