# AI-Guided Discovery and Optimization of Antimicrobial Peptides Through Species-Aware Language Model

## Abstract
The rise of antibiotic-resistant bacteria drives an urgent need for novel antimicrobial agents. Antimicrobial peptides (AMPs) show promise due to their multiple mechanisms of action and reduced propensity for resistance development. Here we present LLAMP (Large Language model for AMP activity prediction), a target species-aware AI model that leverages pre-trained language models to predict minimum inhibitory concentration (MIC) values of AMPs. Through screening approximately 5.5 million peptide sequences, we identified peptides 13 and 16 as the most selective and most potent candidates, respectively. Analysis of attention values allowed us to pinpoint critical amino acid residues (e.g., Trp, Lys, and Phe). Using this information, we enhanced the amphipathicity of peptide 13  through targeted modifications, yielding peptide 13-5 with the highest antimicrobial activity among the variants. Notably, peptides 13-5 and 16 demonstrated antimicrobial potency and selectivity comparable to the clinically investigated AMP pexiganan. Our work demonstrates AI's potential to accelerate peptide-based antibiotic discovery.

## Model weight
You can use peptide tuned ESM-2 model weight here.

https://huggingface.co/Daehun/peptide_tuned_ESM-2

You can download LLAMP model weight here.

https://drive.google.com/file/d/1hmfL7uRZsHo4pn0o0nqaqPcntxGJIhE7/view?usp=sharing

## Model Implementation
The model implemented for the AMP MIC prediction in inference.ipynb.

```
# environment setting
$ git clone https://github.com/GIST-CSBL/LLAMP.git

$ cd LLAMP

$ conda create -n LLAMP python==3.9.13

$ conda activate LLAMP

$ pip install -r requirement.txt

$ pip install ipykernel

$ python -m ipykernel install --user --name LLAMP --display-name "LLAMP"
```

## License

This repository contains materials under two different licenses:

### Code License (PolyForm Noncommercial License 1.0.0)
All source code in this repository is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/), which permits use for non-commercial purposes only.  
See the [LICENSE](LICENSE) file for full terms.

## Contact
Daehun Bae (qoeogns09@gm.gist.ac.kr)

Hojung Nam* (hjnam@gist.ac.kr)

*Corresponding Author

