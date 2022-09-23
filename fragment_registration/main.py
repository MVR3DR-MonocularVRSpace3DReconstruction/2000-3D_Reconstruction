import json
import argparse
import time
import datetime
import os, sys
from os.path import isfile

from open3d_example import check_folder_structure
from initialize_config import initialize_config, dataset_loader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reconstruction system")
    parser.add_argument("--config",
                        help="path to the config file",
                        default="./config.json")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
            initialize_config(config)
            check_folder_structure(config['path_dataset'])
    else:
        # load deafult dataset.
        config = dataset_loader(args.default_dataset)

    times = [0, 0, 0, 0]

    start_time = time.time()
    import make_fragments
    make_fragments.run(config)
    times[0] = time.time() - start_time
    print("====================================")
    print("Fragment Generation Finish")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    input()

    start_time = time.time()
    import register_fragments
    register_fragments.run(config)
    times[1] = time.time() - start_time
    print("====================================")
    print("Registration Finish")
    print("====================================")
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    input()

    start_time = time.time()
    import refine_registration
    refine_registration.run(config)
    times[2] = time.time() - start_time
    print("====================================")
    print("Refine Registration Finish")
    print("====================================")
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    input()

    start_time = time.time()
    import integrate_scene
    integrate_scene.run(config)
    times[3] = time.time() - start_time
    print("====================================")
    print("Integration Finish")
    print("====================================")
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))

    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))