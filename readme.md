This repository contains the code for the WSD api.

# How to run
To build the docker container, place the required model files into ./data. The container requires three files:
	
1. The pretrained sloberta2 model (https://www.clarin.si/repository/xmlui/handle/11356/1397), which should be placed inside ./data/sloberta2
2. A trained wsd model, which should be placed inside ./data and named wsd_model.ckpt
3. A sense inventory. Currently, we use elexis-wsd-sl_sense-inventory.tsv, which should be placed inside ./data
