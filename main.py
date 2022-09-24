import json
import argparse
import time
import datetime
import os, sys
from os.path import isfile

from fragment_registration.open3d_utils import check_folder_structure
from fragment_registration.initialize_config import initialize_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reconstruction system")
    parser.add_argument("--debug_mode",
                        help="turn on debug mode.",
                        action="store_true")
    parser.add_argument("--config",
                        help="path to the config file",
                        default="fragment_registration/config.json")
    parser.add_argument("--path_dataset",
                        default="data/redwood-boardroom/")

    args = parser.parse_args()

    

    assert args.config != None
    with open(args.config) as json_file:
        config = json.load(json_file)
        config['path_dataset'] = args.path_dataset
        initialize_config(config)
        check_folder_structure(config['path_dataset'])


    if args.debug_mode:
        config['debug_mode'] = True
    else:
        config['debug_mode'] = False



    times = [0, 0, 0, 0, 0]

    start_time = time.time()
    import fragment_registration.make_fragments
    fragment_registration.make_fragments.run(config)
    times[0] = time.time() - start_time
    print("====================================")
    print("Fragment Generation Finish")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    # print("# Press \"Enter\" to continue.")
    # input()

    start_time = time.time()
    # import method_Append
    # method_Append.run(config)
    times[0] = time.time() - start_time
    print("====================================")
    print("Coarse Fragments Append")
    print("====================================")
    print("- Append fragments    %s" % datetime.timedelta(seconds=times[1]))
    # print("# Press \"Enter\" to continue.")
    # input()

    start_time = time.time()
    import fragment_registration.register_fragments
    fragment_registration.register_fragments.run(config)
    times[1] = time.time() - start_time
    print("====================================")
    print("Registration Finish")
    print("====================================")
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[2]))
    # print("# Press \"Enter\" to continue.")
    # input()

    start_time = time.time()
    import fragment_registration.refine_registration
    fragment_registration.refine_registration.run(config)
    times[2] = time.time() - start_time
    print("====================================")
    print("Refine Registration Finish")
    print("====================================")
    print("- Refine registration %s" % datetime.timedelta(seconds=times[3]))
    # print("# Press \"Enter\" to continue.")
    # input()

    start_time = time.time()
    import fragment_registration.integrate_scene
    fragment_registration.integrate_scene.run(config)
    times[3] = time.time() - start_time
    print("====================================")
    print("Integration Finish")
    print("====================================")
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[4]))

    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))