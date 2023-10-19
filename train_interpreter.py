# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Generate images from seeds, prepare features and targets, and train interpeter
Generate annotated dataset using generator and interpreter
Details in https://arxiv.org/pdf/2104.06490.pdf
"""

import re
from typing import List, Optional
import click
from tqdm import trange
from pathlib import Path
from datasetgan_utils import *


#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''
    Accept either a comma separated list of numbers 'a,b,c' 
    or a range 'a-c' and return as a list of ints.
    '''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', required=True,
              help='Network pickle filename')
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, default=1,
              show_default=True, help='Truncation psi')
@click.option('--class', 'class_idx', type=int,
              help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode',
              type=click.Choice(['const', 'random', 'none']),
              default='const', show_default=True)
@click.option('--result-dir', type=str, required=True, metavar='DIR',
              help='Path where checkpoints and logs will be stored')
@click.option('--annotation-file', type=str, required=True, metavar='FILE',
              help='Path to annotations')
@click.option('--n', 'num_samples', type=int, default=1000,
              help='Number of annotated samples to generate')
@click.option('--sv', 'save_dir', type=str, required=True, metavar='DIR',
              help='Path where generated dataset will be saved')
@click.option('--mc', 'memory_constrained', type=int, default=1,
              show_default=True,
              help='Use less features in memory-constrained environments')
@click.option('--train-only', is_flag=True,
              help='Option to train interpreter without generating dataset')
@click.option('--multiclass', is_flag=True,
              help='Option to train interpreter with multiclass labels')
@click.option('--epochs', type=int, default=2, show_default=True,
              help='Number of epochs used to train the interpreter')
@click.option('--batch-size', type=int, default=32, show_default=True,
              help='Batch size used to train the interpreter')
@click.option('--batch-gen', type=int, default=0, show_default=True,
              help="Batch size used for the generation")
@click.option('--verbose', '-v', is_flag=True, help="Print more information")
def main(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    result_dir: str,
    annotation_file: str,
    num_samples: int,
    save_dir: str,
    class_idx: Optional[int],
    memory_constrained: Optional[int],
    train_only: Optional[bool],
    multiclass: Optional[bool],
    epochs: Optional[int],
    batch_size: Optional[int],
    batch_gen: Optional[int],
    verbose: Optional[bool],
):

    # Parse command-line arguments and load objects into memory
    train_seeds, annotations, augmented_generator, device, label, multiclass = \
        initial_setup(verbose, seeds, class_idx, network_pkl, annotation_file,
                      multiclass, result_dir, save_dir)
    
    # Prepare Training Data for Interpreter
    X, y, palette = prepare_training_data(train_seeds, annotations,
                                          augmented_generator, device, label,
                                          verbose, truncation_psi, noise_mode,
                                          memory_constrained, result_dir,
                                          multiclass)
    
    # Train Interpreter
    interpreter = train_interpreter(X, y, multiclass=multiclass,
                                    batch_size=batch_size, max_epochs=epochs)
    print(f"Trained interpreter! Sample annotations in: {result_dir}")
    if train_only:
        print("Leaving ...")
        return
    
    # Generate Synthetic Annotated Dataset
    if batch_gen:
        # Generate in chunks
        batch_size = batch_gen
        for seed in trange(0, num_samples, batch_size):
            seeds = list(range(seed, min(seed+batch_size, num_samples)))
            generate_labeled_images(seeds, augmented_generator, interpreter, 
                                    device, label, verbose, truncation_psi,
                                    noise_mode, memory_constrained, save_dir,
                                    result_dir, palette, num_samples,
                                    train_seeds)
    else:
        # Generate one image at a time
        for seed in range(num_samples):
            generate_labeled_images([seed], augmented_generator, interpreter, 
                                    device, label, verbose, truncation_psi,
                                    noise_mode, memory_constrained, save_dir,
                                    result_dir, palette, num_samples,
                                    train_seeds)
    print(f"Generated dataset at: {save_dir}!")
    print("Leaving ...")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
