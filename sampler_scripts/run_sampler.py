import argparse
import yaml
import os
from cosmicfishpie.likelihood import NautilusSampler

# Parse command line args
parser = argparse.ArgumentParser(description="CosmicFish Pie Likelihood Module")
parser.add_argument("config_file", help="Path to YAML config file")
parser.add_argument("--pool", type=int, default=None, help="Number of CPU cores to use for parallel sampling (default: None)")
parser.add_argument("--name", type=str, default=None, help="Base name for output files (default: None)")
args = parser.parse_args()

# Load config
with open(args.config_file, 'r') as f:
    config = yaml.safe_load(f)

# Update config with command line args ONLY if they were explicitly provided
if args.pool is not None:
    config["sampler_settings"]["pool"] = args.pool
if args.name is not None:
    config["name"] = args.name

# Save updated config
updated_config_file = args.config_file.replace('.yaml', '_updated.yaml')
with open(updated_config_file, 'w') as f:
    yaml.dump(config, f)

# Run sampler
sampler = NautilusSampler(config)
sampler.run()