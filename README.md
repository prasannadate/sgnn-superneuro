# sgnn-superneuro
Implementing spiking graph neural networks in SuperNeuro.

Each of the individual datasets have the corresponding configs inside of the `configs` directory
To configure the script to utilize a different config please refer to the this line:
`config = yaml.safe_load(open("configs/citeseer/default_citeseer_config.yaml"))`

To run you simply use: `python3 gnn_citation_networks.py`
